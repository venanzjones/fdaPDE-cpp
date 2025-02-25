'''
#include <gtest/gtest.h> // testing framework
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "src/clustering_test.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
'''
#include <fdaPDE/core.h>
#include <cstddef>
#include <chrono>
#include <unsupported/Eigen/SparseExtra>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono> 
#include <filesystem>
#include <limits>

using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::PDE;
using fdapde::core::reaction;
using fdapde::core::laplacian;

#include "src/utils/constants.h"
#include "src/utils/mesh_loader.h"
#include "src/utils/utils.h"

using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::core::Grid;
using fdapde::core::Triangulation;
#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include "../fdaPDE/models/regression/strpde.h"
#include "../fdaPDE/models/sampling_design.h"
using fdapde::models::STRPDE;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::Sampling;
using fdapde::core::Kronecker;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::spline_order;
using fdapde::core::Kronecker;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::spline_order;
int main()
{
// Discretize space
DMatrix<double> nodes = read_mtx<double>("data/thesis/mesh/nodes.mtx");
DMatrix<int> elements = read_mtx<int>("data/thesis/mesh/elements.mtx");

for(int i = 0; i < elements.rows(); ++i){
    for(int j = 0; j < elements.cols(); ++j){
        elements(i,j) -= 1;
    }
}
Eigen::MatrixXi boundary = Eigen::MatrixXi::Zero(nodes.rows(), 1);
Triangulation<3,3> brain_mesh(nodes, elements, boundary);
auto L = reaction<FEM>(1.0);
DMatrix<double> u = DMatrix<double>::Zero(brain_mesh.n_cells() * 4, 1);
PDE<Triangulation<3,3>, decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(brain_mesh, L, u);
pde.init();
const SpMatrix<double>& R0 = pde.mass(); 
std::cout << "R0: " << R0.rows() << "   " << R0.cols() << std::endl;
// Discretize time
Triangulation<1, 1> time_mesh(0, 0.263157894736842, 128);
auto Lt = reaction<SPLINE>(1.0);
PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_pde(time_mesh, Lt);
time_pde.init();
const SpMatrix<double>& T0 = time_pde.mass();
std::cout << "T0: " << T0.rows() << "   " << T0.cols() << std::endl;
const SpMatrix<double>& R0_tilde = Kronecker(R0, T0);
std::cout << "Kron:"  << R0_tilde.rows() << "   " << R0_tilde.cols() << std::endl;

}