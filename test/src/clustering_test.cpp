#include <fdaPDE/core.h>
#include <gtest/gtest.h>   
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

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"

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
#include "fdaPDE/models/clustering/new_kmeans.h"
#include "fdaPDE/models/clustering/kmedoids.h"
#include "fdaPDE/models/clustering/dbscan.h"
#include "fdaPDE/models/clustering/hierarchical.h"

using fdapde::models::ManualInitPolicy;
using fdapde::models::RandomInitPolicy;
using fdapde::models::KMeans;
using fdapde::models::KppPolicy;
using fdapde::models::HAC;
using fdapde::models::KMedoids;
using fdapde::models::DBSCAN;


// I/O utils 
template <typename T> DMatrix<T> read_mtx(const std::string& file_name) {
    SpMatrix<T> buff;
    Eigen::loadMarket(buff, file_name);
    return buff;
}
template<typename T> void eigen2ext(const DMatrix<T>& M, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app); 
    
    for(long int i = 0; i < M.rows(); ++i) {
            for(long int j=0; j < M.cols()-1; ++j) file << M(i,j) << sep;
            file << M(i, M.cols()-1) <<  "\n";  
    }
    file.close();
}
template<typename T> void eigen2txt(const DMatrix<T>& M, const std::string& filename = "mat.txt", bool append = false){
    eigen2ext<T>(M, " ", filename, append);
}
template<typename T> void eigen2csv(const DMatrix<T>& M, const std::string& filename = "mat.csv", bool append = false){
    eigen2ext<T>(M, ",", filename, append);
}
template<typename T> void vector2ext(const std::vector<T>& V, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app);
    
    for(std::size_t i = 0; i < V.size()-1; ++i) file << V[i] << sep;
    
    file << V[V.size()-1] << "\n";  
    
    file.close();
}
template<typename T> void vector2txt(const std::vector<T>& V, const std::string& filename = "vec.txt", bool append = false){
   vector2ext<T>(V, " ", filename, append);
}
template<typename T> void vector2csv(const std::vector<T>& V, const std::string& filename = "vec.csv", bool append = false){
   vector2ext<T>(V, ",", filename, append);
}
void write_table_noHeaders(const DMatrix<double>& M, const std::string& filename = "data.txt") {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    eigen2txt<double>(M, filename, false);  // Directly write the matrix without any headers
}
double compute_accuracy(const std::vector<int> &memberships,
                        const Eigen::Matrix<int, -1, -1> &ground_truth,
                        const std::vector<int> &relabel_map) {
    int correct = 0;
    for (size_t i = 0; i < memberships.size(); ++i) {
        if (relabel_map[memberships[i]] == ground_truth(i)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / memberships.size();
}

// --------------------------------------------------------------------------------------------------
/*
make
mkdir build
mv fdapde_test build
cd build
./fdapde_test
*/

// Run the HAC experiment for 1D data
void run_hac_experiment_1D(int n, int n_curves) {

    Eigen::MatrixXd Y = read_mtx<double>("../data/models/clustering/1D/y_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
    Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/1D/memberships_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
    ground_truth = ground_truth.array() -1; // subtract one since memberships are (1,...,k) and we need (0,...,k-1)
    int executions = 1;
    int k = 3;
    std::vector<double> execution_times;
    std::vector<double> accuracies;
    execution_times.reserve(executions);
    accuracies.reserve(executions);
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n);
    auto nodes = interval.nodes();

    Eigen::MatrixXd R0 = read_mtx<double>("../data/models/clustering/1D/R0_" + std::to_string(n) + ".mtx");
    for (int ex = 0; ex < executions; ++ex) {
        std::cout << "\n";

        // Measure execution time
        auto start_time = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);
        AverageLinkage<double,std::size_t> myLinkage;
        HAC<L2Policy, AverageLinkage<double,std::size_t>> hac(Y, myDist, myLinkage);
        hac.run();
        std::cout << "Running HAC with Avg Linkage on 1D:" << std::endl;
        std::vector<int> memberships(3*n_curves);
        hac.cut_at_k(k, memberships);
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double,std::milli>(end_time - start_time).count();
        execution_times.push_back(elapsed_time);
        std::vector<std::vector<int>> permutations = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        double best_accuracy = 0.0;
        for (const auto& relabel_map : permutations) {
            best_accuracy = std::max(best_accuracy,
                                    compute_accuracy(memberships, ground_truth, relabel_map));
        }
        accuracies.push_back(best_accuracy);

        std::cout << "Accuracy for n = " << n << ", n_curves = " << 3*n_curves
                << ", execution " << ex + 1 << ": " << best_accuracy * 100.0 << "%" << std::endl;
        std::cout << "Execution time:" << elapsed_time << "ms" << std::endl;

    }

    // Save execution times
    std::ofstream exec_file("../results/exec_1D_" + std::to_string(n) + "_" + std::to_string(n_curves) + "_hac.txt");
    for (const auto& time : execution_times) {
        exec_file << time << "\n";
    }
    exec_file.close();

}



TEST(test_1d, hac)
{   
    std::vector<int> n_values = {10,100,1000};
    std::vector<int> n_curves_values = {10};

    for (int n_curves : n_curves_values) {
        for (int n : n_values) {
            run_hac_experiment_1D(n, n_curves);
        }
    }
}


// TEST: 2D on unit square coarse mesh
TEST(test_2d_kpp, unit_square_coarse)
{
auto mesh_name = "unit_square_coarse";
MeshLoader<Triangulation<2, 2>> domain(mesh_name);
auto L = reaction<FEM>(1.0);
DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
pde.init();
DMatrix<double> R0 = pde.mass();  
std::vector<int> n_curves = {100,500,1000};
int n_sim = 1;
DMatrix<double> exec_times(n_curves.size(), n_sim);
std::string data_path = "../data/models/clustering/2D/";
int curr_row = 0;
for(auto & n : n_curves){
    std::cout << "\n";
    Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/2D/unit_square_coarse_" + std::to_string(n) + "_memberships.mtx");
    ground_truth = ground_truth.array(); // subtract one since memberships are (1,...,k) and we need (0,...,k-1)
    std::string file = data_path + mesh_name + "_" + std::to_string(n) + "_Y" + ".mtx";
    DMatrix<double> Y = read_mtx<double>(file);
    int k = 2;
    DVector<double> v_elapsed;
    v_elapsed.resize(n_sim);
    for(int sim = 0; sim < n_sim; ++sim){
        auto t_start = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);
        KppPolicy<L2Policy> myInit(myDist);
        KMeans<L2Policy, KppPolicy<L2Policy>> km(Y, myDist, myInit, k, 100);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "Running K-means++ on the unit square:" << std::endl;
        auto res = km.memberships();
        std::vector<std::vector<int>> permutations = {{0, 1}, {1,0}};
        double best_accuracy = 0.0;
        for (const auto& relabel_map : permutations) {
            best_accuracy = std::max(best_accuracy,
                                    compute_accuracy(res, ground_truth, relabel_map));
        }
        std::cout << "Accuracy for n_curves = " << n << " on the coarse mesh: " << best_accuracy * 100.0 << "%" << std::endl;
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "Execution time:" << elapsed_ms << "ms" << std::endl;
        v_elapsed(sim) = elapsed_ms;
        write_table_noHeaders(exec_times, "../results/exec_2D_" + std::to_string(n) + "_coarse_kpp.txt");

    }   
    exec_times.row(curr_row) = v_elapsed;
    curr_row++;
}
}
