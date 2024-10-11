#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework
#include <cstddef>

using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::PDE;
using fdapde::core::reaction;
using fdapde::core::laplacian;
using fdapde::core::Mesh;
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
#include "../../fdaPDE/models/model_base.h"
#include "../../fdaPDE/models/clustering/clustering_base.h"
#include "../../fdaPDE/models/clustering/kmeans.h" 
#include "../../fdaPDE/models/clustering/hierarchical_single.h"
#include "../../fdaPDE/models/sampling_design.h"
#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"

using fdapde::models::Sampling;
using fdapde::models::SpaceOnly;
using fdapde::models::ModelBase;
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::models::KMEANS;
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/gcv.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::HierarchicalSingle;

using fdapde::models::SpaceOnly;
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::core::Grid;

TEST(dbscan_test, 1d_test) {

    Mesh<1, 1> unit_interval(0, 6.28318530717959, 5);
    auto L = reaction<SPLINE>(1.0);
    PDE<Mesh<1, 1>, decltype(L), DMatrix<double>, SPLINE, spline_order<3>> pde(unit_interval, L);
    pde.init();
    DMatrix<double> X = read_csv<double>("../data/models/clustering/1D_test/X.csv");
    DMatrix<double> Y = read_csv<double>("../data/models/clustering/1D_test/Y.csv");
    HierarchicalSingle model(pde, Sampling::pointwise,3);
    model.set_spatial_locations(X);
    double lambda = 0.00003173365;
    model.init();
    SpMatrix<double> A = model.Psi().transpose() * model.Psi() + lambda * model.R1();
    Eigen::SparseLU<SpMatrix<double>> invA;
    invA.compute(A);
    DMatrix<double> f_spline = invA.solve(model.Psi().transpose() * Y.transpose()).transpose();
    model.set_data(f_spline);
    model.solve();
}

/*
TEST(dbscan_test, 1d_test) {

    Mesh<1, 1> unit_interval(0, 6.28318530717959, 5);
    auto L = reaction<SPLINE>(1.0);
    PDE<Mesh<1, 1>, decltype(L), DMatrix<double>, SPLINE, spline_order<3>> pde(unit_interval, L);
    pde.init();
    DMatrix<double> X = read_csv<double>("../data/models/clustering/1D_test/X.csv");
    DMatrix<double> Y = read_csv<double>("../data/models/clustering/1D_test/Y.csv");
    DBSCAN model(pde, Sampling::pointwise, 0.1, 3);
    model.set_spatial_locations(X);
    double lambda = 0.00003173365;
    model.init();
    SpMatrix<double> A = model.Psi().transpose() * model.Psi() + lambda * model.R1();
    Eigen::SparseLU<SpMatrix<double>> invA;
    invA.compute(A);
    DMatrix<double> f_spline = invA.solve(model.Psi().transpose() * Y.transpose()).transpose();
    model.set_data(f_spline);
    model.solve();
}


TEST(kmeans_test, clustering_1d_test) {

    Mesh<1, 1> unit_interval(0, 6.28318530717959, 5);
    auto L = reaction<SPLINE>(1.0);
    PDE<Mesh<1, 1>, decltype(L), DMatrix<double>, SPLINE, spline_order<3>> pde(unit_interval, L);
    pde.init();
    DMatrix<double> X = read_csv<double>("../data/models/clustering/1D_test/X.csv");
    DMatrix<double> Y = read_csv<double>("../data/models/clustering/1D_test/Y.csv");
    KMEANS model(pde, Sampling::pointwise); 
    model.set_spatial_locations(X);
    double lambda = 0.00003173365;
    model.init();
    SpMatrix<double> A = model.Psi().transpose() * model.Psi() + lambda * model.R1();
    Eigen::SparseLU<SpMatrix<double>> invA;
    invA.compute(A);
    DMatrix<double> f_spline = invA.solve(model.Psi().transpose() * Y.transpose()).transpose();
    unsigned k = 3; 
    model.set_k(k);
    model.set_data(f_spline.transpose());
    model.solve();
    // EXPECT_TRUE(int_equal(model.memberships(), "../data/models/clustering/1D_test/ground_truth.csv"));
}




TEST(kmeans_test_2, clustering_2d_test) {

    MeshLoader<Mesh2D> domain("unit_square");
    DMatrix<double> X = read_csv<double>("../data/models/clustering/2D_test/X.csv");
    DMatrix<double> Y = read_csv<double>("../data/models/clustering/2D_test/Y.csv");

    auto L_aux = -laplacian<FEM>();
    DMatrix<double> u_aux = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L_aux), DMatrix<double>, FEM, fem_order<1>> pde_aux(domain.mesh, L_aux, u_aux);
    DMatrix<double> f_spline(Y.rows(), domain.mesh.n_nodes());
    double lambda = 0.001;

    for (int i = 0; i < Y.rows(); ++i) 
    { 
        SRPDE aux(pde_aux, Sampling::pointwise);
        Eigen::MatrixXd y_temp = Y.row(i).transpose();
        aux.set_spatial_locations(X);
        BlockFrame<double, int> df;
        df.insert(OBSERVATIONS_BLK, y_temp);
        aux.set_data(df);
        aux.set_lambda_D(lambda);
        aux.init();
        aux.solve();
        f_spline.row(i) = aux.f().transpose();
    }
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();

    unsigned k = 3; 
    KMEANS model(pde, Sampling::mesh_nodes); 
    model.set_k(k);
    model.init();
    model.set_data(f_spline);
    model.solve();
    // EXPECT_TRUE(int_equal(model.memberships(), "../data/models/clustering/1D_test/ground_truth.csv"));
}

TEST(kmeans_test_3, clustering_2d_test_gcv) {

    MeshLoader<Mesh2D> domain("unit_square");
    DMatrix<double> X = read_csv<double>("../data/models/clustering/2D_test/X.csv");
    DMatrix<double> Y = read_csv<double>("../data/models/clustering/2D_test/Y.csv");

    auto L_aux = -laplacian<FEM>();
    DMatrix<double> u_aux = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L_aux), DMatrix<double>, FEM, fem_order<1>> pde_aux(domain.mesh, L_aux, u_aux);
    DMatrix<double> f_spline(Y.rows(), domain.mesh.n_nodes());
    std::size_t seed = 2024;
    DMatrix<double> lambdas(25, 1);
    for (int i = 0; i < 25; ++i) { lambdas(i, 0) = std::pow(10, -3.0 + 0.25 * i); }

    for (int i = 0; i < Y.rows(); ++i) 
    { 
        SRPDE aux(pde_aux, Sampling::pointwise);
        Eigen::MatrixXd y_temp = Y.row(i).transpose();
        aux.set_spatial_locations(X);
        BlockFrame<double, int> df;
        df.insert(OBSERVATIONS_BLK, y_temp);
        aux.set_data(df);
        auto GCV = aux.gcv<StochasticEDF>(100, seed);
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas);
        auto lambda = opt.optimum();
        aux.set_lambda_D(lambda(0));
        aux.init();
        aux.solve();
        f_spline.row(i) = aux.f().transpose();
    }
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();

    unsigned k = 3; 
    KMEANS model(pde, Sampling::mesh_nodes); 
    model.set_k(k);
    model.init();
    model.set_data(f_spline);
    model.solve();
    // EXPECT_TRUE(int_equal(model.memberships(), "../data/models/clustering/1D_test/ground_truth.csv"));
}
*/