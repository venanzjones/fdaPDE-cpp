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

#ifndef LINKAGES_H
#define LINKAGES_H

#include <Eigen/Dense>

template <typename T1, typename T2>
struct SingleLinkage {
    double update_distance(const T1 dist_ic, 
                           const T1 dist_jc, 
                           const T2 /*size_i*/,
                           const T2 /*size_c*/) const 
    {
        return std::min(dist_ic, dist_jc);
    }
};

template <typename T1, typename T2>
struct CompleteLinkage {
    double update_distance(const T1 dist_ic, 
                           const T1 dist_jc, 
                           const T2 /*size_i*/,
                           const T2 /*size_c*/) const 
    {
        return std::max(dist_ic, dist_jc);
    }
};

template <typename T1, typename T2>
struct AverageLinkage {
    double update_distance(const T1 dist_ic, 
                           const T1 dist_jc, 
                           const T2 size_i,
                           const T2 size_j) const 
    {
        return (dist_ic * size_i + dist_jc * size_j) 
               / static_cast<T1>(size_i + size_j);
    }
};

#endif // LINKAGES_H
