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
/*
void write_table(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.txt"){

    std::ofstream file(filename);

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2txt<std::string>(head, filename);    
    }else vector2txt<std::string>(header, filename);
    
    eigen2txt<double>(M, filename, true);
}
*/
void write_table_noHeaders(const DMatrix<double>& M, const std::string& filename = "data.txt") {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    eigen2txt<double>(M, filename, false);  // Directly write the matrix without any headers
}
/*
void write_csv(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.csv"){
    std::ofstream file(filename);

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2csv(head, filename);    
    }else vector2csv(header, filename);
    
    eigen2csv<double>(M, filename, true);
}*/
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
// km: 99.6667%
// 

// --------------------------------------------------------------------------------------------------
/*
make
mkdir build
mv fdapde_test build
cd build
./fdapde_test
cd ..
rm build -r
*/
// --------------------------------------------------------------------------------------------------


void run_experiment_1D(int n, int n_curves) {

        Eigen::MatrixXd Y = read_mtx<double>("../data/models/clustering/1D/y_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
        Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/1D/memberships_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
        ground_truth = ground_truth.array() -1;
        int executions = 1;
        int k = 3;

        std::vector<double> execution_times;
        // std::vector<std::vector<int>> matrix_seeds;
        std::vector<double> accuracies;

        execution_times.reserve(executions);
        // matrix_seeds.reserve(executions);
        accuracies.reserve(executions);
        Triangulation<1, 1> interval(0, 2*std::numbers::pi, n);
        auto nodes= interval.nodes();
        /*
        Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n, n);
        // Loop over all elements to compute contributions
        for (int i = 0; i < n - 1; ++i) {
            // Element length h
            double h = nodes(i + 1) - nodes(i);
            double k_local[2][2] = {
                {2.0 / 6 * h, 2.0 / 6 * h},
                {1.0 / 6 * h, 1.0 / 6 * h}
            };

            // Assemble into global stiffness matrix
            R0(i, i)     += k_local[0][0];
            R0(i, i + 1) += k_local[0][1];
            R0(i + 1, i) += k_local[1][0];
            R0(i + 1, i + 1) += k_local[1][1];
        }*/
        Eigen::MatrixXd R0 = read_mtx<double>("../data/models/clustering/1D/R0_" + std::to_string(n) + ".mtx");
        for (int ex = 0; ex < executions; ++ex) {

            // Measure execution time
            auto start_time = std::chrono::high_resolution_clock::now();
            L2Policy myDist(R0);
            AverageLinkage<double,std::size_t> myLinkage;
            HAC<L2Policy, AverageLinkage<double,std::size_t>> hac(Y, myDist, myLinkage);
            hac.run();
            std::vector<int> memberships(3*n_curves);
            hac.cut_at_k(k, memberships);
            auto end_time = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration<double,std::milli>(end_time - start_time).count();
            execution_times.push_back(elapsed_time);

            // for( auto &m : memberships) std::cout << m << " ";
            std::vector<std::vector<int>> permutations = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

            double best_accuracy = 0.0;
            for (const auto& relabel_map : permutations) {
                best_accuracy = std::max(best_accuracy,
                                        compute_accuracy(memberships, ground_truth, relabel_map));
            }
            accuracies.push_back(best_accuracy);

            std::cout << "Best accuracy for n = " << n << ", n_curves = " << n_curves
                    << ", execution " << ex + 1 << ": " << best_accuracy * 100.0 << "%" << std::endl;
        }

        // Save execution times
        std::ofstream exec_file("../results/execution_1D_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".txt");
        for (const auto& time : execution_times) {
            exec_file << time << "\n";
        }
        exec_file.close();
        // Save accuracies
        std::ofstream acc_file("../results/accuracy_1D_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".txt");
        for (const auto& acc : accuracies) {
            acc_file << acc << "\n";
        }
        acc_file.close();
    }

    void run_experiment_missing(int n, int n_curves, double p, int missing_pattern = 0) {
    // Read data and ground truth as before
    Eigen::MatrixXd Y = read_mtx<double>("../data/models/clustering/1D/y_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
    Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/1D/memberships_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".mtx");
    ground_truth = ground_truth.array() -1;
    int executions = 1;
    int k = 3;

    std::vector<double> exec_times;
    std::vector<double> accuracies;
    exec_times.reserve(executions);
    accuracies.reserve(executions);

    // Define the domain
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n);
    auto nodes = interval.nodes();
    double L = 2 * std::numbers::pi;

    // (Other precomputed matrices, e.g. R0, are built as before)
    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n - 1; ++i) {
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }

    // Set the random seed (same as before)
    int seed = 2025 + n + n_curves;
    std::mt19937 gen(seed);

    // Apply missing data according to the selected pattern:
    if (missing_pattern == 0) {
        // Pattern 0: Randomly select entries to be missing.
        for (int i = 0; i < Y.rows(); ++i) {
            int n_to_na = n - static_cast<int>(std::round(n * p));
            std::vector<int> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            for (int c = 0; c < n_to_na; ++c) {
                Y(i, indices[c]) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    } else if (missing_pattern == 1) {
        // Pattern 1: Remove values outside a random valid subinterval.
        std::uniform_real_distribution<double> distA(0.0, L / 3.0);
        std::uniform_real_distribution<double> distB(2.0 * L / 3.0, L);
        for (int i = 0; i < Y.rows(); ++i) {
            double a = distA(gen);
            double b = distB(gen);
            for (int j = 0; j < Y.cols(); ++j) {
                double x_j = nodes(j);
                if (x_j < a || x_j > b) {
                    Y(i, j) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    } else if (missing_pattern == 2) {
        // Pattern 2: New pattern – Random holes.
        // For each row (curve) choose m uniformly from {1,2,3,4}
        // and then “remove” (set to NaN) the values in m disjoint subintervals.
        // We want the total missing proportion to be (1-p), so the total length of holes is:
        //   total_missing_length = (1-p)*L.
        // We then set each hole to have fixed length: l = total_missing_length/m.
        // std::uniform_int_distribution<int> m_dist(2, 3);
        for (int i = 0; i < Y.rows(); ++i) {
            // int m = m_dist(gen);
            int m = 1;
            double total_missing_length = (1 - p) * L;
            double l = total_missing_length / m;  // length of each hole

            // To generate m disjoint intervals of length l uniformly in [0,L],
            // sample m numbers uniformly in [0, L - m*l], sort them,
            // then define the j-th hole as:
            //    [ r[j] + j*l,  r[j] + j*l + l ]
            std::vector<double> r(m);
            std::uniform_real_distribution<double> start_dist(0, L - m * l);
            for (int j = 0; j < m; ++j) {
                r[j] = start_dist(gen);
            }
            std::sort(r.begin(), r.end());
            std::vector<std::pair<double, double>> holes;
            for (int j = 0; j < m; ++j) {
                double start = r[j] + j * l;
                double end = start + l;
                holes.emplace_back(start, end);
            }

            // For each column (node), if the node lies inside any hole, mark it as missing.
            for (int j = 0; j < Y.cols(); ++j) {
                double x = nodes(j);
                for (const auto& hole : holes) {
                    if (x >= hole.first && x <= hole.second) {
                        Y(i, j) = std::numeric_limits<double>::quiet_NaN();
                        break; // No need to check further holes.
                    }
                }
            }
        }
    } else {
        std::cerr << "Unknown missing_pattern value." << std::endl;
        return;
    }

    // (Optionally, write out the modified Y for inspection)
    write_table_noHeaders(Y, "../data/models/clustering/1D/y_" + std::to_string(n) + "_" +
                                  std::to_string(n_curves) + "_missing_" + std::to_string(p) + ".txt");

    // Run the clustering experiment (timing, k-means, etc.) as before
    for (int ex = 0; ex < executions; ++ex) {
        auto t_start = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);
        RandomInitPolicy myInit;
        KMeans<L2Policy, RandomInitPolicy> km(Y, myDist, myInit, k, 100, 2024);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        exec_times.push_back(elapsed_ms);

        std::vector<std::vector<int>> permutations = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

        double best_accuracy = 0.0;
        auto memberships = km.memberships();
        for (const auto &relabel_map : permutations) {
            best_accuracy = std::max(best_accuracy, compute_accuracy(memberships, ground_truth, relabel_map));
        }
        accuracies.push_back(best_accuracy);
        std::cout << "Best accuracy for n = " << n << ", n_curves = " << n_curves
                  << ", execution " << ex + 1 << ": " << best_accuracy * 100.0 << "%" << std::endl;
    }

    // Save execution times and accuracies as before
    std::ofstream exec_file("../results/execution_1D_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".txt");
    for (const auto& time : exec_times) {
        exec_file << time << "\n";
    }
    exec_file.close();

    std::ofstream acc_file("../results/accuracy_1D_" + std::to_string(n) + "_" + std::to_string(n_curves) + ".txt");
    for (const auto& acc : accuracies) {
        acc_file << acc << "\n";
    }
    acc_file.close();
}
/*
TEST(test_2d, def_grid)
{
auto mesh_name = "unit_square_fine";
MeshLoader<Triangulation<2, 2>> domain(mesh_name);
auto L = reaction<FEM>(1.0);
DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
pde.init();
DMatrix<double> R0 = pde.mass();  
std::vector<int> n_curves = {100,500,1000};
int n_sim = 1;
DMatrix<double> exec_times(n_curves.size(), n_sim);
std::string data_path = "../data/models/clustering/2D_test/";
int curr_row = 0;
for(auto & n : n_curves){
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
        auto res = km.memberships();
        std::vector<int> cluster_counts(k, 0);
        for (auto label : res) {
            if (label >= 0 && label < k) {
                cluster_counts[label]++;
            }
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << cluster_counts[0] << " " << cluster_counts[1] << "\n";
        v_elapsed(sim) = elapsed_ms;
    }   
    exec_times.row(curr_row) = v_elapsed;
    curr_row++;
}
write_table_noHeaders(exec_times, "../results/2D/km_exec_coarse.txt");
}
*/
/*
TEST(test_2d, def_grid)
{
auto mesh_name = "unit_square";
MeshLoader<Triangulation<2, 2>> domain(mesh_name);
auto L = reaction<FEM>(1.0);
DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
pde.init();
DMatrix<double> R0 = pde.mass();  
std::vector<int> n_curves = {100,500,1000};
int n_sim = 1;
DMatrix<double> exec_times(n_curves.size(), n_sim);
std::string data_path = "../data/models/clustering/2D_test/";
int curr_row = 0;
for(auto & n : n_curves){
    std::string file = data_path + mesh_name + "_" + std::to_string(n) + "_Y" + ".mtx";
    DMatrix<double> Y = read_mtx<double>(file);
    int k = 2;
    DVector<double> v_elapsed;
    v_elapsed.resize(n_sim);
    for(int sim = 0; sim < n_sim; ++sim){
        auto t_start = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);                       
        RandomInitPolicy myInit;
        KMeans<L2Policy, RandomInitPolicy> km(Y, myDist, myInit, k, 100, 2024+sim);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto res = km.memberships();
        std::vector<int> cluster_counts(k, 0);
        for (auto label : res) {
            if (label >= 0 && label < k) {
                cluster_counts[label]++;
            }
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << cluster_counts[0] << " " << cluster_counts[1] << "\n";
        v_elapsed(sim) = elapsed_ms;
    }   
    exec_times.row(curr_row) = v_elapsed;
    curr_row++;
}
write_table_noHeaders(exec_times, "../results/2D/km_random_exec.txt");
}
*/
/*
TEST(test_2d, def_grid)
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
std::string data_path = "../data/models/clustering/2D_test/";
int curr_row = 0;
for(auto & n : n_curves){
    std::string file = data_path + mesh_name + "_" + std::to_string(n) + "_Y" + ".mtx";
    DMatrix<double> Y = read_mtx<double>(file);
    int k = 2;
    DVector<double> v_elapsed;
    v_elapsed.resize(n_sim);
    for(int sim = 0; sim < n_sim; ++sim){
        auto t_start = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);                       
        KMedoids<L2Policy> km(Y, myDist, k, 100);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto res = km.memberships();
        std::vector<int> cluster_counts(k, 0);
        for (auto label : res) {
            if (label >= 0 && label < k) {
                cluster_counts[label]++;
            }
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << cluster_counts[0] << " " << cluster_counts[1] << "\n";
        v_elapsed(sim) = elapsed_ms;
    }   
    exec_times.row(curr_row) = v_elapsed;
    curr_row++;
}
write_table_noHeaders(exec_times, "../results/2D/kmed_coarse_exec.txt");
}
*/
/*
TEST(test_2d, def_grid)
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
std::string data_path = "../data/models/clustering/2D_test/";
int curr_row = 0;
for(auto & n : n_curves){
    std::string file = data_path + mesh_name + "_" + std::to_string(n) + "_Y" + ".mtx";
    DMatrix<double> Y = read_mtx<double>(file);
    int k = 2;
    DVector<double> v_elapsed;
    v_elapsed.resize(n_sim);
    for(int sim = 0; sim < n_sim; ++sim){
        auto t_start = std::chrono::high_resolution_clock::now();
        L2Policy myDist(R0);
        CompleteLinkage<double,std::size_t> myLinkage;
        HAC<L2Policy, CompleteLinkage<double,std::size_t>> hac(Y, myDist, myLinkage);
        hac.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::vector<int> res(3*n);
        hac.cut_at_k(k, res);        
        std::vector<int> cluster_counts(k, 0);
        for (auto label : res) {
            if (label >= 0 && label < k) {
                cluster_counts[label]++;
            }
        }
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << cluster_counts[0] << " " << cluster_counts[1] << "\n";
        v_elapsed(sim) = elapsed_ms;
    }   
    exec_times.row(curr_row) = v_elapsed;
    curr_row++;
}
write_table_noHeaders(exec_times, "../results/2D/hac_coarse_exec.txt");
}
*/

void store_mtx(const Eigen::MatrixXd &matrix, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Matrix Market header for coordinate format
    file << "%%MatrixMarket matrix coordinate real general\n";

    // Count non-zeros
    int rows = matrix.rows();
    int cols = matrix.cols();
    int nnz = 0;

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            // You could use a small tolerance for floating comparisons:
            // if (std::fabs(matrix(i, j)) > 1e-14) ...
            if (matrix(i, j) != 0.0) {
                nnz++;
            }
        }
    }

    // Write dimensions and number of non-zero entries
    file << rows << " " << cols << " " << nnz << "\n";

    // Write each non-zero entry in (row col value) format, 1-based indexing
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            double val = matrix(i, j);
            if (val != 0.0) {
                file << (i + 1) << " " << (j + 1) << " " << val << "\n";
            }
        }
    }

    file.close();
}
void run_experiment_1d_r0(int n) 
{
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n);
    auto nodes= interval.nodes();

    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n, n);
    // Loop over all elements to compute contributions
    for (int i = 0; i < n - 1; ++i) {
        // Element length h
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };

        // Assemble into global stiffness matrix
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }

    // Store R0 in .mtx format
    std::string filename = "../data/models/clustering/1D/R0_" + std::to_string(n) + ".mtx";
    store_mtx(R0, filename);

    std::cout << "R0 matrix for n = " << n 
              << " stored in file: " << filename << std::endl;
}
TEST(r1d, store)
{   
    std::vector<int> n_values = {10,100,1000};
    std::vector<int> n_curves_values = {10};

    for (int n_curves : n_curves_values) {
        for (int n : n_values) {
            run_experiment_1D(n, n_curves);
        }
    }
}
/*
TEST(test_1d, missing)
{   
    std::vector<int> n_values = {100};
    std::vector<int> n_curves_values = {100};

    for (int n_curves : n_curves_values) {
        for (int n : n_values) {
            double p = 1;
            run_experiment_missing(n, n_curves, p, 2);
        }
    }
}
*/

/*
TEST(a, b)
{
    int n_grid_points = 100;
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n_grid_points);
    auto nodes= interval.nodes();
    DMatrix<double> Y = read_mtx<double>("../data/models/clustering/data/y_100_100.mtx");
    Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/data/memberships_100_100.mtx");

    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n_grid_points, n_grid_points);
    // Loop over all elements to compute contributions
    for (int i = 0; i < n_grid_points - 1; ++i) {
        // Element length h
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };

        // Assemble into global stiffness matrix
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }
    int n_sims = 50;
    std::vector<double> exec_times;
    exec_times.reserve(n_sims);
    for(int i = 0; i < n_sims; ++i){
        auto t_start = std::chrono::high_resolution_clock::now();
        int k = 3;
        L2Policy myDist(R0);
        AverageLinkage<double,std::size_t> myLinkage;
        HAC<L2Policy, AverageLinkage<double,std::size_t>> km(Y, myDist, myLinkage);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::vector<int> memberships(30);
        km.cut_at_k(k, memberships);
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        exec_times.push_back(elapsed_ms);
        std::vector<std::vector<int>> permutations = {{3, 1, 2}, {3, 2, 1}, {1, 3, 2}, {1, 2, 3}, {2, 3, 1}, {2, 1, 3}};
        double best_accuracy = 0.0;
        for (const auto &relabel_map : permutations) {
            best_accuracy = std::max(best_accuracy,
                                     compute_accuracy(memberships, ground_truth, relabel_map));
        }

        std::cout << "Best accuracy: " << best_accuracy * 100.0 << "%" << std::endl;
        }
    
    vector2txt(exec_times, "../results/1d/hac_100_100.txt");
}
*/
/*
TEST(test1d, kmeans10x10)
{
    int n_grid_points = 10;
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n_grid_points);
    auto nodes= interval.nodes();
    DMatrix<double> Y = read_mtx<double>("../data/models/clustering/data/y_10_100.mtx");
    Eigen::Matrix<int, -1, -1> ground_truth = read_mtx<int>("../data/models/clustering/data/memberships_10_100.mtx");

    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n_grid_points, n_grid_points);
    // Loop over all elements to compute contributions
    for (int i = 0; i < n_grid_points - 1; ++i) {
        // Element length h
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };

        // Assemble into global stiffness matrix
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }
    int n_sims = 50;
    std::vector<double> exec_times;
    exec_times.reserve(n_sims);
    for(int i = 0; i < n_sims; ++i){
        auto t_start = std::chrono::high_resolution_clock::now();
        int k = 3;
        L2Policy myDist(R0);
        KppPolicy<L2Policy> myInit(myDist);
        KMeans<L2Policy, KppPolicy<L2Policy>> km(Y, myDist, myInit, k, 100);
        km.run();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto memberships = km.memberships();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        exec_times.push_back(elapsed_ms);
        std::vector<std::vector<int>> permutations = {{3, 1, 2}, {3, 2, 1}, {1, 3, 2}, {1, 2, 3}, {2, 3, 1}, {2, 1, 3}};
        double best_accuracy = 0.0;
        for (const auto &relabel_map : permutations) {
            best_accuracy = std::max(best_accuracy,
                                     compute_accuracy(memberships, ground_truth, relabel_map));
        }

        std::cout << "Best accuracy: " << best_accuracy * 100.0 << "%" << std::endl;
        }
    
    // vector2txt(exec_times, "../results/1D/curves_10_100.txt");
}
*/
/*
TEST(test2d, kmedoidcoarse)
{
    MeshLoader<Triangulation<2, 2>> domain("unit_square_fine");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();  
    std::vector<int> n_curves = {10, 50, 100, 200};
    int n_sim = 50;
    DMatrix<double> exec_times(n_curves.size(), n_sim);
    std::string data_path = "../data/models/clustering/data/";
    int curr_row = 0;
    for(auto & n : n_curves){
        std::string file = data_path + "Y_fine_2D_" + std::to_string(n) + ".mtx";
        DMatrix<double> Y = read_mtx<double>(file);
        int k = 3;
        DVector<double> v_elapsed;
        v_elapsed.resize(n_sim);
        for(int sim = 0; sim < n_sim; ++sim){
            auto t_start = std::chrono::high_resolution_clock::now();
            L2Policy myDist(R0);                       
            KMedoids<L2Policy> km(Y, myDist, k, 100);
            km.run();
            auto t_end = std::chrono::high_resolution_clock::now();
            auto res = km.memberships();
            std::vector<int> cluster_counts(k, 0);
            for (auto label : res) {
                if (label >= 0 && label < k) {
                    cluster_counts[label]++;
                }
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << elapsed_ms'\n';
            std::cout << cluster_counts[0] << " " << cluster_counts[1] << " " << cluster_counts[2] << "\n";
            v_elapsed(sim) = elapsed_ms;
        }   
        
        exec_times.row(curr_row) = v_elapsed;
        curr_row++;
    }
    // write_table_noHeaders(exec_times, "../results/2D/kppcoarseunitsquare.txt");
}
*/
/*
TEST(test2d, kppcoarse)
{
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();  
    std::vector<int> n_curves = {10, 50, 100, 200};
    int n_sim = 50;
    DMatrix<double> exec_times(n_curves.size(), n_sim);
    std::string data_path = "../data/models/clustering/data/";
    int curr_row = 0;
    for(auto & n : n_curves){
        std::string file = data_path + "Y_coarse_2D_" + std::to_string(n) + ".mtx";
        DMatrix<double> Y = read_mtx<double>(file);
        int k = 3;
        DVector<double> v_elapsed;
        v_elapsed.resize(n_sim);
        for(int sim = 0; sim < n_sim; ++sim){
            auto t_start = std::chrono::high_resolution_clock::now();
            L2Policy myDist(R0);                       
            KppPolicy<L2Policy> myInit(myDist);      
            KMeans<L2Policy, KppPolicy<L2Policy>> km(Y, myDist, myInit, k, 100);
            km.run();
            auto t_end = std::chrono::high_resolution_clock::now();
            auto res = km.memberships();
            std::vector<int> cluster_counts(k, 0);
            for (auto label : res) {
                if (label >= 0 && label < k) {
                    cluster_counts[label]++;
                }
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << cluster_counts[0] << " " << cluster_counts[1] << " " << cluster_counts[2] << "\n";
            v_elapsed(sim) = elapsed_ms;
        }   
        
        exec_times.row(curr_row) = v_elapsed;
        curr_row++;
    }
    write_table_noHeaders(exec_times, "../results/2D/kppcoarseunitsquare.txt");
}
*/
/*
TEST(test2d, kppmediumunitsquare)
{
    MeshLoader<Triangulation<2, 2>> domain("unit_square_medium");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();  
    std::vector<int> n_curves = {10, 50, 100, 200};
    int n_sim = 50;
    DMatrix<double> exec_times(n_curves.size(), n_sim);
    std::string data_path = "../data/models/clustering/data/";
    int curr_row = 0;
    for(auto & n : n_curves){
        std::string file = data_path + "Y_medium_2D_" + std::to_string(n) + ".mtx";
        DMatrix<double> Y = read_mtx<double>(file);
        int k = 3;
        DVector<double> v_elapsed;
        v_elapsed.resize(n_sim);
        for(int sim = 0; sim < n_sim; ++sim){
            auto t_start = std::chrono::high_resolution_clock::now();
            L2Policy myDist(R0);                       
            KppPolicy<L2Policy> myInit(myDist);      
            KMeans<L2Policy, KppPolicy<L2Policy>> km(Y, myDist, myInit, k, 100);
            km.run();
            auto t_end = std::chrono::high_resolution_clock::now();
            auto res = km.memberships();
            std::vector<int> cluster_counts(k, 0);
            for (auto label : res) {
                if (label >= 0 && label < k) {
                    cluster_counts[label]++;
                }
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << cluster_counts[0] << " " << cluster_counts[1] << " " << cluster_counts[2] << "\n";
            v_elapsed(sim) = elapsed_ms;
        }   
        
        exec_times.row(curr_row) = v_elapsed;
        curr_row++;
    }
    write_table_noHeaders(exec_times, "../results/2D/kppmediumunitsquare.txt");
}
*/
/*
TEST(test2d, kpp)
{
    MeshLoader<Triangulation<2, 2>> domain("unit_square");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();  
    std::vector<int> n_curves = {10, 50, 100, 200};
    int n_sim = 50;
    DMatrix<double> exec_times(n_curves.size(), n_sim);
    std::string data_path = "../data/models/clustering/data/";
    int curr_row = 0;
    for(auto & n : n_curves){
        std::string file = data_path + "Y_2D_" + std::to_string(n) + ".mtx";
        DMatrix<double> Y = read_mtx<double>(file);
        int k = 3;
        DVector<double> v_elapsed;
        v_elapsed.resize(n_sim);
        for(int sim = 0; sim < n_sim; ++sim){
            auto t_start = std::chrono::high_resolution_clock::now();
            L2Policy myDist(R0);                       
            KppPolicy<L2Policy> myInit(myDist);      
            KMeans<L2Policy, KppPolicy<L2Policy>> km(Y, myDist, myInit, k, 100);
            km.run();
            auto t_end = std::chrono::high_resolution_clock::now();
            auto res = km.memberships();
            std::vector<int> cluster_counts(k, 0);
            for (auto label : res) {
                if (label >= 0 && label < k) {
                    cluster_counts[label]++;
                }
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << cluster_counts[0] << " " << cluster_counts[1] << " " << cluster_counts[2] << "\n";
            v_elapsed(sim) = elapsed_ms;
        }   
        
        exec_times.row(curr_row) = v_elapsed;
        curr_row++;
    }
    write_table_noHeaders(exec_times, "../results/2D/kppcunitsquare.txt");
}
*/

/*
TEST(test2d, kppfine)
{
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R1 = pde.stiff();  
    int n_sim = 1;
    std::string data_path = "../data/models/clustering/2D_test/Y_fine_100.mtx";
    DMatrix<double> Y = read_mtx<double>(data_path);
    int k = 2;
    for(int sim = 0; sim < n_sim; ++sim){

        R1Policy myDist(R1);
        KppPolicy<R1Policy> myInit(myDist);
        KMeans<R1Policy, KppPolicy<R1Policy>> km(Y, myDist, myInit, k, 100);
        km.run();
        auto memberships = km.memberships();
        std::vector<int> cluster_counts(k, 0);
        for (auto label : memberships){
            if (label >= 0 && label < k){
                cluster_counts[label]++;
            }
        }
        for(auto &c : cluster_counts) std::cout << c << " ";
    }   

}// write_table_noHeaders(exec_times, "../results/2D/kppcunitsquarefine.txt");
*/
/*
TEST(kmeans2d, data)
{
    // Initialize the domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_medium");    
    DMatrix<double> Y = read_mtx<double>("../data/models/clustering/data/Y_gaussian_2D.mtx");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();
    std::cout << R0.rows() << " " << R0.cols() << "\n";
    L2Similarity myDist(R0);
    KppPolicy<L2Similarity> myInit(myDist);
    std::cout << Y.rows() << " " << Y.cols() << "\n";

    KMeans<L2Similarity, KppPolicy<L2Similarity>> km(Y, myDist, myInit, 3, 100);
    km.run();
    auto res = km.memberships();
    int k = 3;
    std::vector<int> cluster_counts(k, 0);
    for (auto l : res) {
        if (l >= 0 && l < k) {  // Ensure the label is valid
            cluster_counts[l]++;
        }
    }
    std::cout << "\nCluster counts:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": " << cluster_counts[i] << " points\n";
    }
}
*/
/*
TEST(kmeans2d, data)
{
    // Initialize the domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");    
    DMatrix<double> Y = read_mtx<double>("../data/models/clustering/data/Y_gaussian_2D.mtx");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();
    std::cout << R0.rows() << " " << R0.cols() << "\n";
    L2Similarity myDist(R0);
    KppPolicy<L2Similarity> myInit(myDist);
    std::cout << Y.rows() << " " << Y.cols() << "\n";

    KMeans<L2Similarity, KppPolicy<L2Similarity>> km(Y, myDist, myInit, 3, 100);
    km.run();
    auto res = km.memberships();
    int k = 3;
    std::vector<int> cluster_counts(k, 0);
    for (auto l : res) {
        if (l >= 0 && l < k) {  // Ensure the label is valid
            cluster_counts[l]++;
        }
    }
    std::cout << "\nCluster counts:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": " << cluster_counts[i] << " points\n";
    }
}
*/
/*
TEST(kmeans1d, data)
{
    int n_grid_points = 100;
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n_grid_points);
    auto nodes= interval.nodes();
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_100_1000.mtx");
    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n_grid_points, n_grid_points);
    // Loop over all elements to compute contributions
    for (int i = 0; i < n_grid_points - 1; ++i) {
        // Element length h
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };

        // Assemble into global stiffness matrix
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }
    int k = 3;
    L2Similarity myDist(R0);
    KppPolicy<L2Similarity> myInit(myDist);
    KMeans<L2Similarity, KppPolicy<L2Similarity>> km(Y, myDist, myInit, 3, 100);
    km.run();
    auto res = km.memberships();
    std::vector<int> cluster_counts(k, 0);
    for (auto l : res) {
        if (l >= 0 && l < k) {  // Ensure the label is valid
            cluster_counts[l]++;
        }
    }
    std::cout << "\nCluster counts:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": " << cluster_counts[i] << " points\n";
    }
}
*/
/*
TEST(test,data)
{
    // Initialize the domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square");    
    DMatrix<double> Y = read_mtx<double>("../data/models/clustering/data/Y_shuffled_n441_ncurves10.mtx");
    auto L = reaction<FEM>(1.0);
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
    pde.init();
    DMatrix<double> R0 = pde.mass();
    L2Similarity myDist(R0);
    KppPolicy<L2Similarity> myInit(myDist);
    KMeans<L2Similarity, KppPolicy<L2Similarity>> km(Y, myDist, myInit, 3, 100);
    km.run();
    auto labels = km.memberships();
    for(auto l : labels){
            std::cout << l << "\n";}
}
*/

/*
TEST(kmedoid, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_1000_100.mtx");
    DMatrix<double> X = read_mtx<double>("data/models/clustering/data/x_1000.mtx");
    FEM1DSimilarity myDist(X);
    int k = 3;
    KMedoids<FEM1DSimilarity> km(Y, myDist, k, 100);
    km.run();
    auto res = km.memberships();
        
    std::vector<int> cluster_counts(k, 0);
    for (auto l : res) {
        if (l >= 0 && l < k) {  // Ensure the label is valid
            cluster_counts[l]++;
        }
    }
    std::cout << "\nCluster counts:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << ": " << cluster_counts[i] << " points\n";
    }
}
*/
/*
TEST(test_slink, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_100_10.mtx");
    DMatrix<double> X = read_mtx<double>("data/models/clustering/data/x_100.mtx");
    FEM1DSimilarity myDist(X);
    

    SLINK<FEM1DSimilarity> s(Y, myDist);
    s.run();
    auto res = s.getClusters(3);
    for(auto l : res){
        std::cout << l << "\n";}
}

TEST(test_dbscan, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_100_10.mtx");
    DMatrix<double> X = read_mtx<double>("data/models/clustering/data/x_100.mtx");
    FEM1DSimilarity myDist(X);
    double eps = 1;
    unsigned minPts = 2;

    fdapde::models::DBSCAN<FEM1DSimilarity> dbscan(Y, myDist, eps, minPts);
    dbscan.fit();
    std::cout << "Silhouette score: " << dbscan.silhouette() << "\n";
}
*/
/*
TEST(test, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_100_10.mtx");
    DMatrix<double> X = read_mtx<double>("data/models/clustering/data/x_100_10.mtx");
    std::cout << X << "\n";

    DVector<double> X_row = X.row(0);
    FEM1DSimilarity myDist(X_row);
    KppPolicy<FEM1DSimilarity> myInit(myDist);
    KMeans<FEM1DSimilarity, KppPolicy<FEM1DSimilarity>> km(Y, myDist, myInit, 3, 100);
    km.run();
    auto labels = km.memberships();
    for(auto l : labels){
            std::cout << l << "\n";}

}
*/
/*
TEST(test,data)
{
    MeshLoader<Triangulation<2,2>> domain("unit_square");
    auto f = [](Eigen::VectorXd x) -> double {
        return std::sin(x[0]) * std::cos(x[1]);
    }
    int n_nodes = domain.mesh.n_nodes();
    Eigen::VectorXd Y(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        Y[i] = f(domain.mesh.node(i));
    }
}
*/
/*
TEST(test, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_100_10.mtx");
    std::cout << "\n=== SLINK Single Linkage ===\n";
    {
        SLINK<FEM1DSimilarity> slink(Y, FEM1DSimilarity(2*std::numbers::pi));
        slink.run();
        auto labels = slink.getClusters(3);
        for(auto l : labels){
             std::cout << l << "\n";}
    }

}
*/
/*
TEST(test, data)
{
    DMatrix<double> Y = read_mtx<double>("data/models/clustering/data/y_10_50.mtx");
    fdapde::models::KMeans Kmeans(Y);
    kmeans.setK(3);
    kmeans.setMaxIter(1000);
    auto start = std::chrono::high_resolution_clock::now();
    kmeans.solveTrapezoidal();
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Solving took " << duration.count() << " seconds " << std::endl;
    unsigned iterations = kmeans.getNIter();
    std::cout << "K-Means finished after " << iterations << " iterations." << std::endl;   

   auto memberships = kmeans.getMemberships();

Eigen::saveMarket(memberships,"data/models/clustering/results/results_10_50.mtx");
}
*/
/*
TEST(kmeans1d, data)
{
    int n_grid_points = 100;
    Triangulation<1, 1> interval(0, 2*std::numbers::pi, n_grid_points);
    auto nodes= interval.nodes();
    Eigen::MatrixXd R0 = Eigen::MatrixXd::Zero(n_grid_points, n_grid_points);
    // Loop over all elements to compute contributions
    for (int i = 0; i < n_grid_points - 1; ++i) {
        // Element length h
        double h = nodes(i + 1) - nodes(i);
        double k_local[2][2] = {
            {2.0 / 6 * h, 2.0 / 6 * h},
            {1.0 / 6 * h, 1.0 / 6 * h}
        };

        // Assemble into global stiffness matrix
        R0(i, i)     += k_local[0][0];
        R0(i, i + 1) += k_local[0][1];
        R0(i + 1, i) += k_local[1][0];
        R0(i + 1, i + 1) += k_local[1][1];
    }
    DMatrix<double> f(n_grid_points, 1);
    for (int i = 0; i < n_grid_points; ++i) {
        double x = nodes(i);
        f(i, 0) = std::sin(x/2);
    }
    // Compute the discrete L2 norm: f^T R0 f
    double discrete_l2_norm = std::sqrt((f.transpose() * R0 * f)(0, 0));
    double true_l2_norm = std::sqrt(std::numbers::pi);
    // Compare the norms
    std::cout << "Discrete L2 Norm (f^T R0 f): " << discrete_l2_norm << "\n";
    std::cout << "True L2 Norm: " << true_l2_norm << "\n";
    // Compute relative error
    double relative_error = std::abs(discrete_l2_norm - true_l2_norm) / true_l2_norm * 100.0;
    std::cout << "Relative Error: " << relative_error << "%\n";
  
}
*/
