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

#ifndef __HIERARCHICAL_SINGLE_H__
#define __HIERARCHICAL_SINGLE_H__

#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <limits>
#include "clustering_base.h"

namespace fdapde {
namespace models {

class HierarchicalSingle : public ClusteringBase<HierarchicalSingle, SpaceOnly> {
private:
    typedef ClusteringBase<HierarchicalSingle, SpaceOnly> Base;

public:
    IMPORT_MODEL_SYMBOLS
    using Base::memberships_;
    using Base::n_basis;    // number of spatial basis
    using RegularizationType = SpaceOnly;
    using This = HierarchicalSingle;
    using Base::df_;

    HierarchicalSingle() = default;
    HierarchicalSingle(const Base::PDE& pde, Sampling s, unsigned k)
        : Base(pde, s), k(k) { }

    int n_obs() const { return y().rows(); }   // number of (active) observations
    const DVector<int>& get_memberships() { return memberships_; }  // memberships values

    void solve() {
        int n = n_obs();

        // POINTER REPRESENTATION: SLINK ALGORITHM O(n^2)
        Eigen::VectorXd lambda(n);
        Eigen::VectorXi pi(n);
        Eigen::VectorXd mu(n);

        pi[0] = 0;
        lambda[0] = std::numeric_limits<double>::infinity();

        for (int i = 1; i < n; i++) {

            pi[i] = i;
            lambda[i] = std::numeric_limits<double>::infinity();

            for (int j = 0; j < i; j++) {
                mu[j] = norm2(y().row(i), y().row(n));
            }

            for (int j = 0; j < i; j++) {
                if (lambda[j] >= mu[j]) {
                    mu[pi[j]] = std::min(mu[pi[j]], lambda[j]);
                    lambda[j] = mu[j];
                    pi[j] = i;
                }
                else {
                    mu[pi[j]] = std::min(mu[pi[j]], mu[j]);
                }
            }

            for (int j = 0; j < i; j++) {
                if (lambda[j] >= lambda[pi[j]]) {
                    pi[j] = i;
                }
            }
        }

        // convert the pointer representation to cluster representation:
        // TODO
        // ...

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
    virtual ~HierarchicalSingle() = default;
};

}  // namespace models
}  // namespace fdapde

#endif  // __HIERARCHICAL_SINGLE_H__

