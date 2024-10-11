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

#ifndef __DBSCAN_H__
#define __DBSCAN_H__

#include <vector>
#include <queue>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include "clustering_base.h"
#include "../../core/fdaPDE/geometry/kd_tree.h"

// questo è ok ma la distance euclidea viene calcolate in KDtree, devo sovrascribere la classe?
// per metterci R0?
namespace fdapde {
namespace models {

class DBSCAN : public ClusteringBase<DBSCAN, SpaceOnly> {
private:
    typedef ClusteringBase<DBSCAN, SpaceOnly> Base;

public:
    IMPORT_MODEL_SYMBOLS
    using Base::memberships_;
    using Base::n_basis;    
    using RegularizationType = SpaceOnly;
    using This = DBSCAN;
    using Base::df_;

    double eps;
    unsigned minPts;
    size_t niter_ = 0;

    // constructor
    DBSCAN() = default;
    DBSCAN(const Base::PDE& pde, Sampling s, double eps, unsigned minPts) 
        : Base(pde, s), eps(eps), minPts(minPts) { }

    int n_obs() const { return y().rows(); }   // number of (active) observations
    const DVector<int>& get_memberships() { return memberships_; }  // memberships values

    void solve() {
        // Initialize KDTree with the data points
    
        fdapde::core::KDTree<n_basis> tree(y());

        unsigned clusterId = 0;
        memberships_ = DVector<int>::Constant(n_obs(), -1);  // -1 indicates unvisited points

        for (int i = 0; i < n_obs(); ++i) {
            if (memberships_(i) != -1) continue;  // skip already processed points

            auto neighbors = tree.range_search({y().row(i).array() - eps, y().row(i).array() + eps});
            if (neighbors.size() < minPts) {
                memberships_(i) = 0;  // mark as noise
                continue;
            }

            ++clusterId;
            std::queue<int> q;
            for (auto neighbor : neighbors) {
                memberships_(neighbor) = clusterId;
                q.push(neighbor);
            }

            while (!q.empty()) {
                int current = q.front();
                q.pop();

                auto neighbors2 = tree.range_search({y().row(current).array() - eps, y().row(current).array() + eps});
                if (neighbors2.size() >= minPts) {
                    for (auto neighbor2 : neighbors2) {
                        if (memberships_(neighbor2) == -1 || memberships_(neighbor2) == 0) {
                            memberships_(neighbor2) = clusterId;
                            q.push(neighbor2);
                        }
                    }
                }
            }
        }

        std::cout << "Number of clusters: " << clusterId << std::endl;
        std::cout << "Memberships: " << memberships_.transpose() << std::endl;
    }

    // getters
    virtual ~DBSCAN() = default;
};

}  // namespace models
}  // namespace fdapde

#endif  // __DBSCAN_H__
