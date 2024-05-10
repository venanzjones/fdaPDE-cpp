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

#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/core.h>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/density_estimation/depde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::DEPDE;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(depde_test, test1) {
    // define domain 
    MeshLoader<Triangulation<2, 2>> domain("square_density");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda = 0.1;
    DEPDE<SpaceOnly> model(problem);
    model.set_lambda_D(lambda);

    DMatrix<double> locs   = read_csv<double>("../data/models/depde/2D_test1/data.csv");
    DVector<double> f_init = read_csv<double>("../data/models/depde/2D_test1/f_init.csv");
    model.set_spatial_locations(locs);    
    model.init();
    
    fdapde::core::BFGS<fdapde::Dynamic> opt(500, 1e-5, 1e-2);
    opt.optimize(model, f_init.array().log());
    
    std::cout << opt.optimum() << std::endl;
}
