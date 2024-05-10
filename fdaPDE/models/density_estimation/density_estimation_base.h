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

#ifndef __DENSITY_ESTIMATION_BASE_H__
#define __DENSITY_ESTIMATION_BASE_H__

#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>
#include "../model_macros.h"
#include "../model_traits.h"
#include "../space_only_base.h"
#include "../space_time_separable_base.h"
#include "../sampling_design.h"
using fdapde::core::BinaryMatrix;

namespace fdapde {
namespace models {

template <typename Model, typename RegularizationType>
class DensityEstimationBase :
    public select_regularization_base<Model, RegularizationType>::type,
    public SamplingBase<Model> {
   protected:
    std::function<double(const DVector<double>&)> int_exp_;   // \int_{\mathcal{D}} exp(g(x)), x \in \mathcal{D}
    std::function<DVector<double>(const DVector<double>&)> grad_int_exp_;   // gradient of int_exp
    BinaryMatrix<Dynamic> point_pattern_;   // n_spatial_locs X n_temporal_locs point pattern matrix
    DVector<double> g_;                     // expansion coefficients vector of estimated density function
    DMatrix<double> PsiQuad_;               // reference element basis evaluation at quadrature nodes
    DMatrix<double> w_;                     // quadrature weights
    SpMatrix<double> Upsilon_;              // \Upsilon_(i,:) = \Phi(i,:) \kron S(p_i) \Psi
   public:
    using Base = typename select_regularization_base<Model, RegularizationType>::type;
    using SamplingBase<Model>::Psi;     // matrix of basis evaluations at data locations

    DensityEstimationBase() = default;
    // space-only constructor
    template <typename PDE_>
    DensityEstimationBase(PDE_&& pde)
        requires(is_space_only<Model>::value)
        : Base(pde), SamplingBase<Model>(Sampling::pointwise) {
        pde.init();   // early PDE initialization
        static constexpr int LocalDim = std::decay_t<PDE_>::SpaceDomainType::local_dim;
        static constexpr int n_quadrature_nodes = 6; // -------------------------- generalize wrt dimensionality
	// allocate space
	w_.resize(n_quadrature_nodes, 1);
	PsiQuad_.resize(n_quadrature_nodes, pde.reference_basis().size());
	// compute reference basis evaluation at quadrature nodes
	core::IntegratorTable<LocalDim, n_quadrature_nodes> integrator {};
        for (int i = 0; i < n_quadrature_nodes; ++i) {
            w_(i, 0) = integrator.weights[i];
            for (int j = 0; j < pde.reference_basis().size(); ++j) {
                PsiQuad_(i, j) = pde.reference_basis()[j](integrator.nodes[i]);
            }
        }
        // store functor for approximation of \int_{\mathcal{D}} \exp(g). Computes
        // \sum_{e \in mesh} {e.measure() * \sum_j {w_j * exp[\sum_i {g_i * \psi_i(q_j)}]}}
        int_exp_ = [&](const DVector<double>& g) {
            double result = 0;
            for (auto e = pde.domain().cells_begin(); e != pde.domain().cells_end(); ++e) {
                result +=
                  (w_.transpose() * (PsiQuad_ * g(pde.dofs().row(e->id()))).array().exp().matrix())[0] * e->measure();
            }
            return result;
        };
	// store functor for computation of \nabla_g \int_{\mathcal{D}} \exp(g)
        grad_int_exp_ = [&](const DVector<double>& g) {
            DVector<double> grad(g.rows());
            grad.setZero();
            for (auto e = pde.domain().cells_begin(); e != pde.domain().cells_end(); ++e) {
                grad(pde.dofs().row(e->id())) +=
                  PsiQuad_.transpose() *
                  (PsiQuad_ * g(pde.dofs().row(e->id()))).array().exp().cwiseProduct(w_.array()).matrix() *
                  e->measure();
            }
            return grad;
        };
    }
    // space-time separable constructor
    template <typename SpacePDE_, typename TimePDE_>
    DensityEstimationBase(SpacePDE_&& space_penalty, TimePDE_&& time_penalty)
        requires(is_space_time_separable<Model>::value)
        : Base(space_penalty, time_penalty), SamplingBase<Model>(Sampling::pointwise) {
        // store space-time integrator
    }

    // getters
    int n_obs() const { return point_pattern_.count(); };
    const SpMatrix<double>& Psi() const { return Psi(not_nan()); }
    const SpMatrix<double>& Upsilon() const { return is_space_only<Model>::value ? Psi() : Upsilon_; }
    const DMatrix<double>& PsiQuad() const { return PsiQuad_; }
    const DMatrix<double>& w() const { return w_; }
    double int_exp(const DVector<double>& g) const { return int_exp_(g); }
    double int_exp() const { return int_exp_(g_); }
    DVector<double> grad_int_exp(const DVector<double>& g) const { return grad_int_exp_(g); }
    DVector<double> grad_int_exp() const { return grad_int_exp_(g_); }
    const DVector<double>& g() const { return g_; }          // expansion coefficient vector of log density field
    DVector<double> f() const { return g_.array().exp(); }   // expansion coefficient vector of density field

    // initialization methods
    void analyze_data() {
        point_pattern_ = BinaryMatrix<Dynamic>::Ones(Base::n_locs(), 1);   // ------------- generalize for space-time
        if constexpr (is_space_time_separable<Model>::value) {
            Upsilon_ = point_pattern_.vector_view().repeat(1, Base::n_basis()).select(Psi());
        }
        return;
    }
    void correct_psi() { return; }
    void set_point_pattern(const DMatrix<double>& point_pattern) { point_pattern_ = point_pattern; }
};

}   // namespace models
}   // namespace fdapde

#endif   // __DENSITY_ESTIMATION_BASE_H__
