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

#ifndef __MESH_LOADER_H__
#define __MESH_LOADER_H__

#include <fdaPDE/geometry.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework
#include <random>
using fdapde::core::CSVReader;

namespace fdapde {
namespace testing {

const std::string MESH_PATH = "../data/mesh/";

using MESH_TYPE_LIST = ::testing::Types<
  core::Triangulation<2,2>/*, core::Triangulation<2,3>, core::Triangulation<3,3>*//*, core::NetworkMesh*/>; 

// selects sample mesh depending on the dimensionality of the problem
//     * 1.5D: 204  2D points, 559   elements, 559   edges. /test/data/mesh/linear_newtwork/*.csv
//     * 2D:   3600 2D points, 6962  elements, 10561 edges. /test/data/mesh/unit_square/*.csv
//     * 2.5D: 340  3D points, 616   elements, 956   edges. /test/data/mesh/surface/*.csv
//     * 3D:   587  3D points, 2775  elements, 5795  faces. /test/data/mesh/unit_sphere/*.csv
constexpr auto standard_mesh_selector(unsigned int M, unsigned int N) {
    // first order meshes
    if (M == 1 && N == 2) return "network";       // 1.5D
    if (M == 2 && N == 2) return "unit_square";   // 2D
    if (M == 2 && N == 3) return "surface";       // 2.5D
    if (M == 3 && N == 3) return "unit_sphere";   // 3D
    return "";                                    // error case
}

// An utility class to help in the import of sample test meshes from files
template <typename MeshType_> struct MeshLoader {
    using MeshType = MeshType_;
    static constexpr int M = MeshType::local_dim;
    static constexpr int N = MeshType::embed_dim;
    static constexpr bool manifold = core::is_manifold<M, N>::value;
    MeshType mesh;
    std::random_device rng;
    CSVReader<double> double_reader {};
    CSVReader<int> int_reader;
    DMatrix<double> points_{};
    DMatrix<int> elements_{}, edges_{}, boundary_{};
    typename std::conditional<!core::is_network<M, N>::value, DMatrix<int>, SpMatrix<int>>::type neighbors_;

    // constructors
    MeshLoader(const std::string& meshID) {
        // read data from files
        std::string points_file    = MESH_PATH + meshID + "/points.csv";
        std::string edges_file     = MESH_PATH + meshID + "/edges.csv";
        std::string elements_file  = MESH_PATH + meshID + "/elements.csv";
        std::string neighbors_file = MESH_PATH + meshID + "/neigh.csv";
        std::string boundary_file  = MESH_PATH + meshID + "/boundary.csv";

        points_ = double_reader.parse_file<Eigen::Dense>(points_file);
        // realign indexes to 0, if requested
        elements_ = (int_reader.parse_file<Eigen::Dense>(elements_file).array() - 1).matrix();
        edges_ = (int_reader.parse_file<Eigen::Dense>(edges_file).array() > 0)
                   .select(int_reader.parse_file<Eigen::Dense>(edges_file).array() - 1, -1)
                   .matrix();
        boundary_ = int_reader.parse_file<Eigen::Dense>(boundary_file);
        if constexpr (!core::is_network<M, N>::value)
            neighbors_ = (int_reader.parse_file<Eigen::Dense>(neighbors_file).array() > 0)
                           .select(int_reader.parse_file<Eigen::Dense>(neighbors_file).array() - 1, -1)
                           .matrix();
        else { neighbors_ = int_reader.parse_file<Eigen::Sparse>(neighbors_file); }
        // initialize mesh
        mesh = MeshType(points_, elements_, boundary_);
    }
    // load default mesh according to dimensionality
    MeshLoader() : MeshLoader(standard_mesh_selector(MeshType::local_dim, MeshType::embed_dim)) {};
    // generate element at random inside mesh m
    typename MeshType::CellType generate_random_element() {
        std::uniform_int_distribution<int> random_id(0, mesh.n_cells() - 1);
        int id = random_id(rng);
        return mesh.cell(id);
    };
    // generate point at random inside element e
    SVector<N> generate_random_point(const typename MeshType::CellType& e) {
        std::uniform_real_distribution<double> T(0, 1);
        // let t, s, u ~ U(0,1) and P1, P2, P3, P4 a set of points, observe that:
        //     * if P1 and P2 are the vertices of a linear element, p = t*P1 + (1-t)*P2 lies into it for any t ~ U(0,1)
        //     * if P1, P2, P3 are vertices of a triangle, the point P = (1-t)P1 + t((1-s)P2 + sP3) is in the triangle
        //       for any choice of t, s ~ U(0,1)
        //     * if P1, P2, P3, P4 are vertices of a tetrahedron, then letting Q = (1-t)P1 + t((1-s)P2 + sP3) and
        //       P = (1-u)P4 + uQ, P belongs to the tetrahedron for any choice of t, s, u ~ U(0,1)
        double t = T(rng);
        SVector<N> p = t * e.node(0) + (1 - t) * e.node(1);
        for (int j = 1; j < M; ++j) {
            t = T(rng);
            p = (1 - t) * e.node(1 + j) + t * p;
        }
        return p;
    }
    // generate randomly n pairs <ID, point> on mesh, such that point is contained in the element with identifier ID
    std::vector<std::pair<int, SVector<N>>> sample(int n) {
        // preallocate memory
        std::vector<std::pair<int, SVector<N>>> result {};
        result.resize(n);
        // generate sample
        for (int i = 0; i < n; ++i) {
            auto e = generate_random_element();
            result[i] = std::make_pair(e.id(), generate_random_point(e));
        }
        return result;
    }
};

}   // namespace testing
}   // namespace fdapde

#endif   // __MESH_LOADER_H__
