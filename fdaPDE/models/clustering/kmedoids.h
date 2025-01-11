// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef KMEDOIDS_H
#define KMEDOIDS_H

#include "dissimilarities.h" 
#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>

inline constexpr unsigned MAX_KMEDOIDS_ITERATIONS = 100; // TODO: is it the best way?

namespace fdapde {
namespace models {

template <typename DistancePolicy>
class KMedoids {

private:
    const Eigen::MatrixXd& Y_;      // n_obs_ x n_nodes matrix 
    DistancePolicy dist_;           // policy for computing distances  
    unsigned k_;                    // number of clusters
    unsigned max_iter_;             // max iterations allowed
    unsigned n_iter_ = 0;           // iterations done
    std::size_t n_obs_;             // number of observations
    Eigen::MatrixXd d_matrix_;      // n_obs_ x n_obs_ distance matrix (precomputed)
    std::vector<int> memberships_;  // memberships vector
    std::vector<int> medoids_;      // vector storing the medoids id
    std::vector<bool> medoid_mask_; // boolean vector to control if an obs is a medoid
    Eigen::VectorXd d_to_medoids_;  // distances to medoids

public:  
    // Constructor
    KMedoids(const Eigen::MatrixXd& Y,      // reference to the data matrix
             const DistancePolicy& dist,    // reference to the distance policy
             int k,                         // number of clusters
             unsigned max_iter = MAX_KMEDOIDS_ITERATIONS)  // max number of iterations
        : Y_(Y),
          dist_(dist),
          k_(k),
          max_iter_(max_iter),
          memberships_(Y.rows(), -1),       // initialize memberships to -1
          medoids_(k, -1),                  // initialize medoids to -1
          medoid_mask_(Y.rows(), false)     // initialize medoid_mask_ to false
    {
        n_obs_ = Y_.rows(); 
        if (k <= 0 || static_cast<std::size_t>(k) > n_obs_) { 
            throw std::runtime_error("Invalid number of clusters.");
        }
        // Assembly the NxN distance matrix (O(n_obs_^2))
        d_matrix_.resize(n_obs_, n_obs_);
        for (std::size_t i = 0; i < n_obs_; ++i) {
            d_matrix_(i, i) = 0.0;
            for (std::size_t j = i + 1; j < n_obs_; ++j) {
                double d = dist_(Y_.row(i), Y_.row(j));
                d_matrix_(i, j) = d;
                d_matrix_(j, i) = d;
            }
        }
        d_to_medoids_.resize(n_obs_);
    }

    // Main routine, applies the Partitioning Around Medoids (PAM) algorithm, O(k*n^2)
    void run() {
        if (!n_obs_) {
            throw std::runtime_error("No data provided.");
        }

        build();                  // Initialize medoids, O(k*n^2)
        bool f_improved = true;
        while(f_improved && n_iter_ < max_iter_) {
            assign_obs();         // Assign observations to medoids, O(k*n)
            f_improved = swap();  // SWAP procedure, O(k*n^2)
            ++n_iter_;
        }

        std::cout << " Execution completed in " << n_iter_
                  << " iterations (max=" << max_iter_ << ").\n";
    }

    // Getters
    const std::vector<int>& memberships() const { return memberships_; } 
    const std::vector<int>& medoid_ids() const { return medoids_; }      
    const Eigen::VectorXd& d_to_medoids() const { return d_to_medoids_; }  
    unsigned n_iterations() const { return n_iter_; }                    

private:

    // BUILD procedure
    void build() {
        Eigen::VectorXd dist_to_set(n_obs_);
        dist_to_set.setConstant(std::numeric_limits<double>::infinity());

        for (int m = 0; m < k_; ++m) {
            std::size_t best_candidate(0); 
            double best_cost = std::numeric_limits<double>::infinity();

            // Loop over all the observations
            for (std::size_t c = 0; c < n_obs_; ++c) {
                if (medoid_mask_[c]) { // Skip if already a medoid
                    continue;
                }

                // Compute the cost of c and update eventually
                Eigen::VectorXd col_c = d_matrix_.col(c);
                Eigen::VectorXd merged = dist_to_set.cwiseMin(col_c);
                double cost_c = merged.sum();
                if (cost_c < best_cost) {
                    best_cost = cost_c;
                    best_candidate = c;
                }
            }

            // Take the best candidate as medoid and set the mask
            medoids_[m] = best_candidate;
            medoid_mask_[best_candidate] = true;

            // Update dist_to_set
            Eigen::VectorXd col_best = d_matrix_.col(best_candidate);
            dist_to_set = dist_to_set.cwiseMin(col_best);
        }
    }

    // Assign each observation to its closest medoid
    void assign_obs() {
        // Loop over the observations
        for (std::size_t i = 0; i < n_obs_; ++i) {
            double best_d = std::numeric_limits<double>::infinity();
            int best_m = -1;
            // Loop over all the medoids
            for (int m = 0; m < k_; ++m) {
                int medoid_idx = medoids_[m];
                double d = d_matrix_(i, medoid_idx); 
                if (d < best_d) {
                    best_d = d;
                    best_m = m;
                }
            }
            // Assign membership and distances (needed in swap())
            memberships_[i] = best_m;
            d_to_medoids_[i] = best_d;
        }
    }

    // SWAP procedure 
    bool swap() {

        double best_delta = 0.0; 
        int best_medoid_position = -1;
        std::size_t best_non_medoid = -1;

        // Loop over the medoids
        for (int m = 0; m < k_; ++m) {
            int old_medoid = medoids_[m];
            for (std::size_t c = 0; c < n_obs_; ++c) {
                if (medoid_mask_[c]) {
                    continue; // Skip if c is already a medoid
                }
                // What happens if we swap m and c?
                double delta = swap_delta(m, old_medoid, c); 
                if (delta < best_delta) {
                    best_delta = delta;
                    best_medoid_position = m; 
                    best_non_medoid = c;
                }
            }
        }
        if (best_delta < 0.0 && best_medoid_position >= 0) {
            // do the swap
            int old_idx = medoids_[best_medoid_position];
            medoid_mask_[old_idx] = false;
            medoids_[best_medoid_position] = best_non_medoid;
            medoid_mask_[best_non_medoid] = true;
            return true;
        }
        return false;
    }

    // Computes the cost of swapping old_medoid with candidate
    double swap_delta(int cluster_id,           // cluster id (0,...,k-1)
                      int old_medoid,           // old medoid id 
                      std::size_t candidate) const // candidate id 
    {
        double delta = 0.0;
        // Loop over the observations
        for (std::size_t i = 0; i < n_obs_; ++i) {
            double old_cost = d_to_medoids_[i];  // Current distance to the assigned medoid
            double new_cost = old_cost;          

            // Check if i's assigned medoid was old_medoid
            int i_cluster = memberships_[i];
            int i_med = medoids_[i_cluster];
            if (i_med == old_medoid) {
                double best_d = std::numeric_limits<double>::infinity();
                // Loop over the clusters
                for (int l = 0; l < k_; ++l) {
                    // if l == cluster_id, we are considering the candidate as medoid
                    if (l == cluster_id) {
                        double d_c = d_matrix_(i, candidate);
                        if (d_c < best_d) best_d = d_c;
                    } else { // otherwise, we keep the old medoid
                        int row_m = medoids_[l];
                        double d_m = d_matrix_(i, row_m);
                        if (d_m < best_d) best_d = d_m;
                    }
                }
                new_cost = best_d;
            } else {
                // See if candidate is closer
                double d_c = d_matrix_(i, candidate);
                if (d_c < old_cost) {
                    new_cost = d_c;
                }
            }
            // Add the contribution of observation i 
            delta += (new_cost - old_cost);
        }
        return delta;
    }
};

} // namespace models
} // namespace fdapde

#endif // KMEDOIDS_H


