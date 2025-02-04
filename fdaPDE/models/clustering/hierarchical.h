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

#ifndef HAC_H
#define HAC_H

#include "linkages.h"
#include "dissimilarities.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace fdapde {
namespace models {

template <typename DistancePolicy, typename LinkagePolicy>
class HAC {
private:
    const Eigen::MatrixXd& Y_;
    std::size_t n_obs_;
    
    DistancePolicy dist_;
    LinkagePolicy linkage_;

    Eigen::MatrixXd dist_matrix_;

   
    std::vector<int> merges_;  // Merge step t -> store the pair of merged items on merges_[2*t], merges_[2*t + 1] 
    std::vector<double> heights_; // Store the "height" (i.e. distance) at each merge step

public:
    // Constructor
    HAC(const Eigen::MatrixXd& Y,
        const DistancePolicy& dist,
        const LinkagePolicy& linkage)
        : Y_(Y),
          n_obs_(Y.rows()),
          dist_(dist),
          linkage_(linkage),
          dist_matrix_(Y.rows(), Y.rows()),
          merges_(2 * (Y.rows() - 1), 0),
          heights_(Y.rows() - 1, 0.0)
    {
        if (n_obs_ < 2) {
            throw std::runtime_error("Provide at least two observations");
        }
        // Precompute the distance matrix, O(n^2) memory/time
        for (std::size_t i = 0; i < n_obs_; ++i) {
            dist_matrix_(i, i) = 0.0;
            for (std::size_t j = i + 1; j < n_obs_; ++j) {
                double d = dist_(Y_.row(i), Y_.row(j));
                dist_matrix_(i, j) = d;
                dist_matrix_(j, i) = d;
            }
        }
    }

    // Main routine
    void run() {
        std::vector<int> cluster_id(n_obs_);
        for (std::size_t i = 0; i < n_obs_; ++i) {
            cluster_id[i] = i+1;
        }

        std::vector<bool> active(n_obs_, true); // Bool vector to store the active clusters
        std::vector<double> cluster_size(2 * n_obs_ - 1, 1.0);

        int next_cluster_id = static_cast<int>(n_obs_ + 1);

        for (std::size_t step = 0; step < n_obs_ - 1; ++step) {
            double min_dist = std::numeric_limits<double>::infinity();
            int ci = -1;
            int cj = -1;

            // Find the closest pair
            for (std::size_t i = 0; i < n_obs_; ++i) {
                if (!active[i]) continue;
                for (std::size_t j = i + 1; j < n_obs_; ++j) {
                    if (!active[j]) continue;
                    double d = dist_matrix_(i, j);
                    if (d < min_dist) {
                        min_dist = d;
                        ci = static_cast<int>(i);
                        cj = static_cast<int>(j);
                    }
                }
            }
            heights_[step] = min_dist;

            // Negative -> single obs, positive -> previously formed cluster (id-n_obs_)
            int leftID  = cluster_id[ci];
            int rightID = cluster_id[cj];
            int leftVal  = (leftID  <= static_cast<int>(n_obs_)) ? -leftID  : (leftID  - static_cast<int>(n_obs_));
            int rightVal = (rightID <= static_cast<int>(n_obs_)) ? -rightID : (rightID - static_cast<int>(n_obs_));

            merges_[2 * step + 0] = leftVal;
            merges_[2 * step + 1] = rightVal;

            // Merge clusters ci and cj 
            active[cj] = false;
            cluster_id[ci] = next_cluster_id;
            cluster_id[cj] = -1;
            cluster_size[next_cluster_id - 1] =
                cluster_size[leftID  - 1] +
                cluster_size[rightID - 1];

            // Update distances to the new cluster
            for (std::size_t k = 0; k < n_obs_; ++k) {
                if (!active[k] || k == static_cast<std::size_t>(ci)) {
                    continue;
                }
                double dist_ic = dist_matrix_(ci, k);
                double dist_jc = dist_matrix_(cj, k);

                double size_i = cluster_size[leftID  - 1];
                double size_j = cluster_size[rightID - 1];

                double new_dist = linkage_.update_distance(dist_ic, dist_jc, size_i, size_j);
                dist_matrix_(ci, k) = new_dist;
                dist_matrix_(k, ci) = new_dist;
            }
            ++next_cluster_id;
        }
    }

    // Splits into k clusters by skipping the last (n_obs_-k), for more references:
    // https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/cutree
    void cut_at_k(unsigned k, std::vector<int>& labels) const {
        labels.resize(n_obs_);
        if (k > n_obs_ || k < 2) {
            std::fill(labels.begin(), labels.end(), 0);
            return;
        }
        unsigned merges_to_use = static_cast<unsigned>(n_obs_ - k);

        std::vector<int> last_merge(n_obs_, 0);

        for (unsigned step = 1; step <= merges_to_use; step++) {
            int m1 = merges_[2 * (step - 1) + 0];
            int m2 = merges_[2 * (step - 1) + 1];

            if (m1 < 0 && m2 < 0) {
                last_merge[-m1 - 1] = step;
                last_merge[-m2 - 1] = step;
            }
            else if (m1 < 0 || m2 < 0) {
                if (m1 < 0) std::swap(m1, m2);
                int single_idx = -m2;
                for (std::size_t l = 0; l < n_obs_; ++l) {
                    if (last_merge[l] == m1) {
                        last_merge[l] = step;
                    }
                }
                last_merge[single_idx - 1] = step;
            }
            else {
                for (std::size_t l = 0; l < n_obs_; ++l) {
                    if (last_merge[l] == m1 || last_merge[l] == m2) {
                        last_merge[l] = step;
                    }
                }
            }
        }

        int label = 0;
        std::vector<int> temp(n_obs_ + 1, -1);
        for (std::size_t j = 0; j < n_obs_; j++) {
            if (last_merge[j] == 0) {
                labels[j] = label++;
            } else {
                if (temp[last_merge[j]] < 0) {
                    temp[last_merge[j]] = label++;
                }
                labels[j] = temp[last_merge[j]];
            }
        }
    }

    // Cuts the dendrogram by an arbitrary distance threshold, 
    // this method relies on cut_at_k()
    void cut_at_distance(double cutoff, std::vector<int>& labels) const {
        unsigned step = 0;
        while (step < n_obs_ - 1 && heights_[step] < cutoff) {
            ++step;
        }
        unsigned k = static_cast<unsigned>(n_obs_ - step);
        cut_at_k(k, labels);
    }

    // Returns the merge heights (size n_obs_-1).
    const std::vector<double>& heights() const {
        return heights_;
    }
};

} // namespace models
} // namespace fdapde

#endif // HAC_H
