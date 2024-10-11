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


#ifndef __CLUSTERING_BASE_H__
#define __CLUSTERING_BASE_H__

#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>
#include "../model_macros.h"
#include "../model_traits.h"
#include "../space_only_base.h"
#include "../sampling_design.h"

using fdapde::core::BinaryVector;

namespace fdapde {
namespace models {

// base class for any *clustering* model

template <typename Model, typename RegularizationType>
class ClusteringBase :
    public select_regularization_base<Model, RegularizationType>::type,
    public SamplingBase<Model> {

   protected:

    // room for problem solution
    DVector<int> memberships_ {};       // final memberships (N x 1 vetor with entries 1,2,...,k)

   public:

    using Base = typename select_regularization_base<Model, RegularizationType>::type;
    using Base::df_;                    // BlockFrame for problem's data storage
    using Base::idx;                    // indices of observations
    using Base::n_basis;                // number of basis function over physical domain
    using Base::R0;                     // mass matrix
    using Base::R1;
    using Base::n_locs;                 // number of locations
    using SamplingBase<Model>::D;       // for areal sampling, matrix of subdomains measures, identity matrix otherwise
    using SamplingBase<Model>::Psi;     // matrix of spatial basis evaluation at locations p_1 ... p_n
    using SamplingBase<Model>::PsiTD;   // block \Psi^\top*D (not nan-corrected)
    using Base::model;

    ClusteringBase() = default;
    // space-only and space-time parabolic constructor (they require only one PDE)
    ClusteringBase(const Base::PDE& pde, Sampling s)
        requires(is_space_only<Model>::value)
        : Base(pde), SamplingBase<Model>(s) {};
    void init() { model().init_sampling(true); }
    // getters

    const DMatrix<double>& x() const { return df_.template get<double>(X_BLK); }            // observation matrix X
    const DMatrix<double>& y() const { return df_.template get<double>(Y_BLK); };           // Y values
    const SpMatrix<double>& Psi() const { return Psi(not_nan()); }

    int n_obs() const { return y().rows(); }   // number ofobservations
};

}   // namespace models
}   // namespace fdapde

#endif   // __CLUSTERING_BASE_H__
