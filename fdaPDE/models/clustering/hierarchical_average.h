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

#ifndef __HIERARCHICAL_AVERAGE_H__
#define __HIERARCHICAL_AVERAGE_H__

#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <limits>
#include "clustering_base.h"

namespace fdapde {
namespace models {

class HierarchicalAverage : public ClusteringBase<HierarchicalAverage, SpaceOnly> {
private:
    typedef ClusteringBase<HierarchicalAverage, SpaceOnly> Base;

public:
    IMPORT_MODEL_SYMBOLS
    using Base::memberships_;
    using Base::n_basis;    // number of spatial basis
    using RegularizationType = SpaceOnly;
    using This = HierarchicalAverage;
    using Base::df_;
    unsigned k;
    HierarchicalAverage() = default;
    HierarchicalAverage(const Base::PDE& pde, Sampling s, unsigned k)
        : Base(pde, s), k(k) { }

    int n_obs() const { return y().rows(); }   // number of (active) observations
    const DVector<int>& get_memberships() { return memberships_; }  // memberships values
    void set_data(const DMatrix<double>& Y) {
        df_.insert(Y_BLK, Y);
    }

    double cluster_distance(const std::vector<int>& cluster1, const std::vector<int>& cluster2) {
        double max_dist = 0.0;
        for (int i : cluster1) {
            for (int j : cluster2) {
                double dist = norm2(y().row(i) , y().row(j));
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
        }
        return max_dist;
    }

    void solve() {
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
    double norm2(const DVector<double> &op1, const DVector<double> &op2) const {
        auto fg = (op1 - op2).transpose() * R0() * (op1 - op2);
        auto f = (op1).transpose() * R0() * (op1);
        auto g = (op2).transpose() * R0() * (op2);
        return fg(0,0) / (f(0,0) + g(0,0)); 
    }
    virtual ~HierarchicalAverage() = default;
};

}  // namespace models
}  // namespace fdapde

#endif  // __HIERARCHICAL_AVERAGE_H__

