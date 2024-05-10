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

#ifndef __DEPDE_H__
#define __DEPDE_H__

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "density_estimation_base.h"

namespace fdapde {
namespace models {

template <typename RegularizationType_>
class DEPDE : public DensityEstimationBase<DEPDE<RegularizationType_>, RegularizationType_> {
   private:
    using Base = DensityEstimationBase<DEPDE<RegularizationType_>, RegularizationType_>;
    double llik_;        // -1^\top * \Upsilon * g
    double int_exp_g_;   // \sum_{e \in mesh} w^\top * exp[\PsiQuad * g_e]
    double pen_;         // g^\top * P_{\lambda_D, \lambda_T} * g
   public:
    using RegularizationType = std::decay_t<RegularizationType_>;
    using This = DEPDE<RegularizationType>;
    using Base::grad_int_exp;   // \nabla_g (\int_{\mathcal{D}} \exp(g))
    using Base::int_exp;        // \int_{\mathcal{D}} \exp(g)
    using Base::n_obs;          // number of observations
    using Base::P;              // discretized penalty matrix
    using Base::PsiQuad;        // reference basis evaluation at quadrature nodes
    using Base::Upsilon;        // \Upsilon_(i,:) = \Phi(i,:) \kron S(p_i) \Psi
    using Base::w;              // weights of quadrature rule

    // space-only constructor
    template <typename PDE_>
    DEPDE(PDE_&& pde) requires(is_space_only<This>::value) : Base(pde) { }
    // space-time separable constructor
    template <typename SpacePDE_, typename TimePDE_>
    DEPDE(SpacePDE_&& space_penalty, TimePDE_&& time_penalty) requires(is_space_time_separable<This>::value)
        : Base(space_penalty, time_penalty) { }

    // evaluates penalized negative log-likelihood at point
    // L(g) = - 1^\top*\Upsilon*g + \sum_{e \in mesh} w^\top*exp[\Psi_q*g_e] + \lambda_S*g^\top*P*g
    double operator()(const DVector<double>& g) {
        return -(Upsilon() * g).sum() + n_obs() * int_exp(g) + g.dot(P() * g);
    }
    // log-likelihood gradient functor
    // \nabla_g(L(g)) = -\Upsilon^\top*1 + n*\sum_{e \in mesh} w*exp[\Psi_q*g_e]*\Psi_q^\top + 2*P*g
    std::function<DVector<double>(const DVector<double>&)> derive() {
        return [this](const DVector<double>& g) -> DVector<double> {
            return -Upsilon().transpose() * DVector<double>::Ones(n_obs()) + n_obs() * grad_int_exp(g) + 2 * P() * g;
        };
    }
    void init_model() { return; }
};

}   // namespace models
}   // namespace fdapde

#endif   // __DEPDE_H__
