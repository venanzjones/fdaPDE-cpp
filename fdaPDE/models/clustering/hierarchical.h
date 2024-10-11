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

#ifndef __HIERARCHICAL_CLUSTERING_H__
#define __HIERARCHICAL_CLUSTERING_H__

#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include "clustering_base.h"

using UVector = Eigen::Matrix<unsigned, Eigen::Dynamic, 1>;

namespace fdapde {
namespace models {

enum class LinkageType {
    SINGLE,
    AVERAGE,
    COMPLETE,
    WARD
};

class HierarchicalClustering : public ClusteringBase<HierarchicalClustering, SpaceOnly> {
private:
    typedef ClusteringBase<HierarchicalClustering, SpaceOnly> Base;

public:
    IMPORT_MODEL_SYMBOLS
    using Base::memberships_;
    using Base::distances_;
    using Base::n_basis;    // number of spatial basis
    using RegularizationType = SpaceOnly;
    using This = HierarchicalClustering;
    using Base::df_;

    unsigned k = 2;  // Number of clusters
    LinkageType linkage_type = LinkageType::SINGLE;

    // Constructor
    HierarchicalClustering() = default;
    HierarchicalClustering(const Base::PDE& pde, Sampling s, unsigned k, LinkageType linkage_type)
        : Base(pde, s), k(k), linkage_type(linkage_type) { }

    int n_obs() const { return y().rows(); }   // number of (active) observations
    const DVector<int>& get_memberships() { return memberships_; }  // memberships values

    // Function to compute the distance between two clusters
    double cluster_distance(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        switch (linkage_type) {
            case LinkageType::SINGLE:
                return single_linkage(cluster1, cluster2);
            case LinkageType::AVERAGE:
                return average_linkage(cluster1, cluster2);
            case LinkageType::COMPLETE:
                return complete_linkage(cluster1, cluster2);
            case LinkageType::WARD:
                return ward_linkage(cluster1, cluster2);
            default:
                throw std::invalid_argument("Invalid linkage type");
        }
    }

    double single_linkage(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        double min_dist = std::numeric_limits<double>::max();
        for (int i : cluster1) {
            for (int j : cluster2) {
                double dist = (y().row(i) - y().row(j)).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
        return min_dist;
    }

    double average_linkage(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        double sum_dist = 0.0;
        for (int i : cluster1) {
            for (int j : cluster2) {
                sum_dist += (y().row(i) - y().row(j)).norm();
            }
        }
        return sum_dist / (cluster1.size() * cluster2.size());
    }

    double complete_linkage(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        double max_dist = 0.0;
        for (int i : cluster1) {
            for (int j : cluster2) {
                double dist = (y().row(i) - y().row(j)).norm();
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
        }
        return max_dist;
    }

    double ward_linkage(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        Eigen::VectorXd mean1 = Eigen::VectorXd::Zero(n_basis());
        Eigen::VectorXd mean2 = Eigen::VectorXd::Zero(n_basis());

        for (int i : cluster1) {
            mean1 += y().row(i);
        }
        mean1 /= cluster1.size();

        for (int j : cluster2) {
            mean2 += y().row(j);
        }
        mean2 /= cluster2.size();

        double sum_squares = 0.0;
        for (int i : cluster1) {
            sum_squares += (y().row(i) - mean1).squaredNorm();
        }
        for (int j : cluster2) {
            sum_squares += (y().row(j) - mean2).squaredNorm();
        }
        return sum_squares;
    }

    void solve() {
        // Initialize each point as its own cluster
        std::vector<std::vector<int>> clusters(n_obs());
        for (int i = 0; i < n_obs(); ++i) {
            clusters[i].push_back(i);
        }

        while (clusters.size() > k) {
            // Find the closest two clusters
            double min_dist = std::numeric_limits<double>::max();
            int idx1 = 0, idx2 = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                for (size_t j = i + 1; j < clusters.size(); ++j) {
                    double dist = cluster_distance(clusters[i], clusters[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        idx1 = i;
                        idx2 = j;
                    }
                }
            }

            // Merge the closest two clusters
            clusters[idx1].insert(clusters[idx1].end(), clusters[idx2].begin(), clusters[idx2].end());
            clusters.erase(clusters.begin() + idx2);
        }

        memberships_ = DVector<int>::Zero(n_obs());
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (int point : clusters[i]) {
                memberships_(point) = i;
            }
        }

        std::cout << "Number of clusters: " << clusters.size() << std::endl;
        std::cout << "Memberships: " << memberships_.transpose() << std::endl;
    }

    // Getters
    virtual ~HierarchicalClustering() = default;
};

}  // namespace models
}  // namespace fdapde

#endif  // __HIERARCHICAL_CLUSTERING_H__
