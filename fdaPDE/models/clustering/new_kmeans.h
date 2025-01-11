#ifndef KMEANS_H
#define KMEANS_H

#include "dissimilarities.h"
#include "init_policies.h"

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

inline constexpr unsigned MAX_KMEANS_ITERATIONS = 100; 

namespace fdapde {
namespace models {

template <typename DistancePolicy, typename InitPolicy>
class KMeans {
private:
    const Eigen::MatrixXd& Y_;
    DistancePolicy dist_;
    InitPolicy init_policy_;

    std::size_t n_obs_;   
    unsigned k_;
    unsigned max_iter_;
    unsigned n_iter_ = 0;

    std::vector<int> memberships_;   // initialize with -1
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
          n_obs_(Y.rows()),            // store once here
          k_(k),
          max_iter_(max_iter),
          memberships_(n_obs_, -1),    // initialize with -1
          centroids_(k, Y.cols())
    {
        if (k_ == 0 || k_ > n_obs_) {
            throw std::runtime_error("Invalid k or data size.");
        }
        centroids_.setZero();
        initial_clusters_.reserve(k_);
    }

    // Main routine
    void run() {
        // 1) Initialize centroids
        initial_clusters_ = init_policy_.init(Y_, centroids_, k_);

        bool f_changed = true;
        for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
            f_changed = false;

            // Assignment step
            for (std::size_t i = 0; i < n_obs_; ++i) {
                double bestDist = std::numeric_limits<double>::max();
                int bestC = memberships_[i];
                auto rowI = Y_.row(i);

                for (unsigned c = 0; c < k_; ++c) {
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
            std::vector<std::size_t> counts(k_, 0);

            for (std::size_t i = 0; i < n_obs_; ++i) {
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
    const Eigen::MatrixXd& centroids()  const   { return centroids_;  }
    unsigned n_iterations() const { return n_iter_; }
};

} // namespace models
} // namespace fdapde

#endif // KMEANS_H
