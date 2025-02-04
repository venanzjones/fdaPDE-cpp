#ifndef KMEANS_H
#define KMEANS_H

#include "dissimilarities.h"
#include "init_policies.h"

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <optional>

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

    std::vector<int> memberships_; 
    Eigen::MatrixXd centroids_;
    std::vector<int> initial_clusters_;
    std::optional<unsigned> seed_;      // Seed for random/kmeans++ policies

public:
    KMeans(const Eigen::MatrixXd& Y,
           const DistancePolicy& dist,
           const InitPolicy& init_policy,
           unsigned k = 3,
           unsigned max_iter = MAX_KMEANS_ITERATIONS,
           std::optional<unsigned> seed = std::nullopt)
        : Y_(Y),
          dist_(dist),
          init_policy_(init_policy),
          n_obs_(Y.rows()),            
          k_(k),
          max_iter_(max_iter),
          memberships_(n_obs_, -1),    // initialize memberships with -1
          centroids_(k, Y.cols()),
          seed_(seed)
    {
        if (k_ == 0 || k_ > n_obs_) {
            throw std::runtime_error("Invalid k or data size.");
        }
        centroids_.setZero();
        initial_clusters_.reserve(k_);
    }

    // Main routine
    void run() {
        // Initialize centroids with the selected policy and
        // check if init_policy_.init can be called with a seed parameter
        if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
        initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
        } else {
        // Otherwise call it without the seed parameter (e.g., for manual policy)
        initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
        }
        bool f_changed = true; // bool to check if memberships changed
        for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
            f_changed = false;

            // Assignment step
            for (std::size_t i = 0; i < n_obs_; ++i) {
                double best_dist = std::numeric_limits<double>::max();
                int best_c = memberships_[i];
                auto f_i = Y_.row(i);

                for (unsigned c = 0; c < k_; ++c) {
                    double d = dist_(f_i, centroids_.row(c));
                    if (d < best_dist) {
                        best_dist = d;
                        best_c = static_cast<int>(c); 
                    }
                }

                if (best_c != memberships_[i]) {
                    memberships_[i] = best_c;
                    f_changed = true;
                }
            }

            // Exit earlier, since if memberships did not change
            // => neither will the centroids, counts, etc.
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

    // Methods to extract memberships, centroids, n_iterations
    const std::vector<int>& memberships() const { return memberships_; }
    const Eigen::MatrixXd& centroids() const { return centroids_;  }
    unsigned n_iterations() const { return n_iter_; }
};

} // namespace models
} // namespace fdapde

#endif // KMEANS_H
