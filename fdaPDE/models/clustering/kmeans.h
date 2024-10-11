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

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "clustering_base.h"

using UVector = Eigen::Matrix<unsigned, Eigen::Dynamic, 1>;

namespace fdapde {
namespace models {

class KMEANS : public ClusteringBase<KMEANS, SpaceOnly> {
   private:
    typedef ClusteringBase<KMEANS, SpaceOnly> Base;

   public:
    IMPORT_MODEL_SYMBOLS
    using Base::memberships_;  
    using Base::n_basis;    // number of spatial basis
    using RegularizationType = SpaceOnly;
    using This = KMEANS;
    using Base::df_;
    DMatrix<double> distances_ {};      // final distances from centroids (N x k matrix)
    DMatrix<double> centroids_ {};      // final centroids (k x grid matrix)
    unsigned k = 3;
    unsigned max_iter = 100; // maximum number of iterations (default to 1000)
    size_t niter_ = 0;            // number of final iterations (<= max iter)

    // constructor

    KMEANS() = default;
    KMEANS(const Base::PDE& pde, Sampling s) : Base(pde, s) { };
    int n_obs() const { return y().rows(); }   // number of (active) observations
    const DVector<int> &  get_memberships() { return memberships_; };// memberships values
    void set_k(const unsigned& n_clusters) { k = n_clusters; }
    /*
    void set_data(const BlockFrame<double, int>& df) {
        df_.insert(Y_BLK, df.get<double>(Y_BLK));
    }
    */
   void set_data(const DMatrix<double>& Y) {
        df_.insert(Y_BLK, Y);
    }

    double norm2(const DVector<double> &op1, const DVector<double> &op2) const {
        auto fg = (op1 - op2).transpose() * R0() * (op1 - op2);
        auto f = (op1).transpose() * R0() * (op1);
        auto g = (op2).transpose() * R0() * (op2);
        return fg(0,0) / (f(0,0) + g(0,0)); 
    }
    const unsigned & k_() { return k; }

    void solve() {

        unsigned iter = 0;
        DVector<double> observationDistances =  DVector<double>::Ones(n_obs());
        DVector<double> oldObservationDistances = DVector<double>::Zero(n_obs());
        DVector<int> observationMemberships = DVector<int>::Ones(n_obs());
        DVector<int> oldObservationMemberships = DVector<int>::Zero(n_obs());
        DMatrix<double> centroids = DMatrix<double>::Zero(k_(), n_basis()); 
        
        bool distanceCondition = true;
        bool membershipCondition = true;
        bool iterationCondition = true;
        DVector<int> centerIndices = DVector<int>::LinSpaced(k_(), 0, n_obs() - 1); 
        double tol = 0.0001;
    
        for (size_t j = 0; j < k_(); ++j) {
            centroids.row(j) = y().row(centerIndices(j));
        }
        
        while (distanceCondition && membershipCondition && iterationCondition) {
            ++iter;
            iterationCondition = iter < max_iter;
            membershipCondition = ((observationMemberships.array() != oldObservationMemberships.array()).any());

            oldObservationDistances = observationDistances;
            oldObservationMemberships = observationMemberships;
            // Assignment step
            for (int j = 0; j < n_obs(); ++j) {
                Eigen::VectorXd temp(k_());
                for (size_t i = 0; i < k_(); ++i) {
                    temp(i) = norm2(y().row(j), centroids.row(i));
                }
                Eigen::Index index;
                observationDistances(j) = temp.minCoeff(&index);
                observationMemberships(j) = index;
            }
            distanceCondition = ((observationDistances.array() - oldObservationDistances.array()).cwiseAbs() > tol * observationDistances.array()).any();
            
            centroids = DMatrix<double>::Zero(k_(), n_basis());
            Eigen::VectorXi counts = Eigen::VectorXi::Zero(k_());
            
            for (int j = 0; j < n_obs(); ++j){
                unsigned clusterIndex = observationMemberships(j);
                centroids.row(clusterIndex) += y().row(j);
                counts(clusterIndex)++;
            }

            for (size_t j = 0; j < k_(); ++j) {
                if (counts(j) > 0) {
                    centroids.row(j) /= counts(j);
            }
        }
        }

    distances_ = observationDistances;      // final distances from centroids (N x k matrix)
    centroids_ = centroids;                
    niter_ = iter;  
    memberships_ = observationMemberships; 
    std::cout << "Number of iterations: " << niter_ << std::endl;
    std::cout << "Memberships: " << memberships_.transpose() << std::endl;
    };
    // getters
    virtual ~KMEANS() = default;
};

}   // namespace models
}   // namespace fdapde

#endif   // _KMEANS_H__
