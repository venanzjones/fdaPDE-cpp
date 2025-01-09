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

#ifndef KMEANS_H
#define KMEANS_H

#include "dissimilarities.h" 
#include "init_policies.h"  

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

inline constexpr unsigned MAX_KMEANS_ITERATIONS = 100; // TODO: is it the best way?

// TODO: check coherence with kmedoids
namespace fdapde{
namespace models{

template <typename DistancePolicy, typename InitPolicy>
class KMeans {
private:
    const Eigen::MatrixXd& Y_; 
    DistancePolicy dist_;
    InitPolicy init_policy_;
    unsigned k_;
    unsigned max_iter_;
    unsigned n_iter_ = 0;

    std::vector<int> memberships_;
    Eigen::MatrixXd centroids_;
    std::vector<int> initial_clusters_;

public:
    KMeans(const Eigen::MatrixXd& Y,
           const DistancePolicy& dist,
           const InitPolicy& init_policy,
           unsigned k = 3,
           unsigned max_iter = MAX_KMEANS_ITERATIONS)
        : Y_(Y),
          dist_(dist),
          init_policy_(init_policy),
          k_(k),
          max_iter_(max_iter),
          memberships_(Y.rows(), -1),
          centroids_(k, Y.cols())
    {
        if (k == 0 || k > Y.rows()) {
            throw std::runtime_error("Invalid k or data size.");
        }
        centroids_.setZero();
        initial_clusters_.reserve(k);
    }

    // Main routine
    // TODO: manca da aggiugnere seed a .init
    void run() {
        const int N = static_cast<int>(Y_.rows());
        // 1) Initialize centroids
        initial_clusters_ = init_policy_.init(Y_, centroids_, k_);

        bool f_changed = true;
        for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
            f_changed = false;

            // Assignment step
            for (int i = 0; i < N; ++i) {
                // compute best cluster
                int bestC = memberships_[i];
                double bestDist = std::numeric_limits<double>::max();
                auto rowI = Y_.row(i);

                for (unsigned c = 0; c < k_; ++c) {
                    // distance from rowI to centroids_.row(c)
                    double d = dist_(rowI, centroids_.row(c));
                    if (d < bestDist) {
                        bestDist = d;
                        bestC = static_cast<int>(c);
                    }
                }

                if (bestC != memberships_[i]) {
                    memberships_[i] = bestC;
                    f_changed = true;
                }
            }

            // Exit earlier, since if memberships did not change, 
            // neither will the centroids, counts, etc.
            if (!f_changed) {
                break;
            }

            // Update step
            centroids_.setZero();
            std::vector<int> counts(k_, 0);

            for (int i = 0; i < N; ++i) {
                int c = memberships_[i];
                centroids_.row(c) += Y_.row(i);
                counts[c]++;
            }
            for (unsigned c = 0; c < k_; ++c) {
                if (counts[c] > 0) {
                    centroids_.row(c) /= double(counts[c]);
                }
            }
        }
        /*
        std::cout << " Execution completed in " << n_iter_
                  << " iterations (max=" << max_iter_ << ").\n";
        */
    }

    // Accessors
    const std::vector<int>& memberships() const { return memberships_; } 
    const Eigen::MatrixXd& centroids() const { return centroids_; } 
    unsigned n_iterations() const { return n_iter_; }
};
} // namespace models
} // namespace fdapde

#endif // KMEANS_H