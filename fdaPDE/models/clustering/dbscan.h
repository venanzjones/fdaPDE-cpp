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

#ifndef DBSCAN_H
#define DBSCAN_H

#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

namespace fdapde {
namespace models {

template <typename DistancePolicy>
class DBSCAN {
private:

    using NeighbouroodType = std::vector<int>;
    const Eigen::MatrixXd& Y_;
    DistancePolicy dist_;
    double eps_;
    unsigned min_pts_; 
    std::vector<int> memberships_; // -2: unvisited, -1: noise, >=0: cluster id
    Eigen::MatrixXd dist_matrix_;
    std::vector<int> cluster_counts_;
    int n_clusters_;
    std::size_t n_obs_;
    std::vector<NeighbouroodType> neighborhoods_;

public:

    DBSCAN(const Eigen::MatrixXd& Y,
           const DistancePolicy& dist,
           double eps,
           unsigned min_pts)
        : Y_(Y),
          dist_(dist),
          eps_(eps),
          min_pts_(min_pts),
          memberships_(Y.rows(), -2),
          dist_matrix_(Y.rows(), Y.rows())
    {
        n_obs_ = Y_.rows();
        neighborhoods_.resize(n_obs_);      
        if (Y_.rows() == 0) {
            throw std::runtime_error("Y bust be non-empty");
        }
        if (eps_ <= 0.0) {
            throw std::runtime_error("epsilon must be positive");
        }
        if (min_pts_ <= 1) {
            throw std::runtime_error("min_pts must be > 1");
        }
        for (std::size_t i = 0; i < n_obs_; ++i) {
            dist_matrix_(i, i) = 0.0;
            neighborhoods_[i] = nn_search(i);
            for (std::size_t j = i + 1; j < n_obs_; ++j) {
                double d = dist_(Y_.row(i), Y_.row(j));
                dist_matrix_(i, j) = d;
                dist_matrix_(j, i) = d;
            }
        }
   
    }
    
    void run() {

        n_clusters_ = 0;
        cluster_counts_.clear();

        for (std::size_t i = 0; i < n_obs_; ++i) {
            if (memberships_[i] != -2) {
                continue; // Skip visited points
            }

            const auto& neighbors = neighborhoods_[i];
            if (neighbors.size() < min_pts_) {
                memberships_[i] = -1; // Mark as noise
            } else {
                cluster_counts_.push_back(0); // Add a new cluster
                expand(i, n_clusters_);
                n_clusters_++;
            }
        }
    }

    const std::vector<int>& memberships() const { return memberships_; }
    int num_clusters() const { return n_clusters_; }

private:
    
    NeighbouroodType nn_search(std::size_t i) const {
        NeighbouroodType neighbors;
        for (std::size_t j = 0; j < n_obs_; ++j) {
            if (dist_matrix_(i, j) <= eps_) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    }

    void expand(std::size_t i, int cluster_id) {
    memberships_[i] = cluster_id;
    cluster_counts_[cluster_id]++;

    std::queue<int> q;
    for (int nb : neighborhoods_[i]) {
        q.push(nb);
    }

    while (!q.empty()) {
        int curr = q.front();
        q.pop();

        if (memberships_[curr] == -2) { // If unvisited
            memberships_[curr] = cluster_id;
            cluster_counts_[cluster_id]++;

            // Add neighbors of curr to the queue if it's a core point
            if (neighborhoods_[curr].size() >= min_pts_) {
                for (int nb : neighborhoods_[curr]) {
                    if (memberships_[nb] == -2 || memberships_[nb] == -1) {
                        q.push(nb);
                    }
                }
            }
        } 
        // If noise, include in the cluster
        else if (memberships_[curr] == -1) {
            memberships_[curr] = cluster_id;
            cluster_counts_[cluster_id]++;
        }
    }
}
};

} // namespace models
} // namespace fdapde

#endif // DBSCAN_H
