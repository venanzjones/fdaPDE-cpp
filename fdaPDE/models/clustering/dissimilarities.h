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

#ifndef DISSIMILARITIES_H
#define DISSIMILARITIES_H

#include <Eigen/Dense>
#include <cmath>

// d(f,g) = sqrt((f-g)^T R0 (f-g))
struct L2Policy {
    Eigen::MatrixXd R0_;
    L2Policy(const Eigen::MatrixXd& R0)
        : R0_(R0)
    {
        if (R0_.rows() != R0_.cols()) {
            throw std::runtime_error("R0 must be a square matrix!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const 
    {
        Eigen::VectorXd diff = f - g;
        double squared_norm = diff.transpose() * R0_ * diff;
        return std::sqrt(squared_norm);
    }
};

// d(f,g) = sqrt((f-g)^T R0 (f-g)) / ( sqrt(f^T R0 f) + sqrt(g^T R0 g) )
struct L2NormalizedPolicy {
    Eigen::MatrixXd R0_;

    L2NormalizedPolicy(const Eigen::MatrixXd& R0)
        : R0_(R0)
    {
        if (R0_.rows() != R0_.cols()) {
            throw std::runtime_error("R0 must be a square matrix!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const 
    {
        Eigen::VectorXd diff = f - g;
        double squared_norm = diff.transpose() * R0_ * diff;
        double f_squared = f.transpose() * R0_ * f;
        double g_squared = g.transpose() * R0_ * g;
        double denom = std::sqrt(f_squared) + std::sqrt(g_squared);

        if (denom < 1e-14) {
            return squared_norm; 
        }
        return std::sqrt(squared_norm) / denom;
    }
};

// d(f,g) = sqrt((f-g)^T R1 (f-g))
struct R1Policy {
    Eigen::MatrixXd R1_;

    R1Policy(const Eigen::MatrixXd& R1)
        : R1_(R1)
    {
        if (R1_.rows() != R1_.cols()) {
            throw std::runtime_error("R1 must be a square matrix!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const 
    {
        Eigen::VectorXd diff = f - g;
        double squared_norm = diff.transpose() * R1_ * diff;
        return std::sqrt(squared_norm);
    }
};

// d(f,g) = sqrt((f-g)^T R1 (f-g)) / ( sqrt(f^T R1 f) + sqrt(g^T R1 g) )
struct NormalizedR1Policy {
    Eigen::MatrixXd R1_;

    NormalizedR1Policy(const Eigen::MatrixXd& R1)
        : R1_(R1)
    {
        if (R1_.rows() != R1_.cols()) {
            throw std::runtime_error("R1 must be a square matrix!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const
    {
        Eigen::VectorXd diff = f - g;
        double squared_norm = diff.transpose() * R1_ * diff;
        double f_squared = f.transpose() * R1_ * f;
        double g_squared = g.transpose() * R1_ * g;
        double denom = std::sqrt(f_squared) + std::sqrt(g_squared);

        if (denom < 1e-14) {
            return squared_norm; 
        }
        return std::sqrt(squared_norm) / denom;
    }
};

// d(f,g) = sqrt((f-g)^T (R0 + R1) (f-g))
struct SobolevPolicy {
    Eigen::MatrixXd R0_;
    Eigen::MatrixXd R1_;

    SobolevPolicy(const Eigen::MatrixXd& R0, const Eigen::MatrixXd& R1)
        : R0_(R0), R1_(R1)
    {
        if (R0_.rows() != R0_.cols()) {
            throw std::runtime_error("R0 must be a square matrix!");
        }
        if (R1_.rows() != R1_.cols()) {
            throw std::runtime_error("R1 must be square!");
        }
        if (R0_.rows() != R1_.rows()) {
            throw std::runtime_error("R0 and R1 must have the same dimension!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const 
    {
        Eigen::VectorXd diff = f - g;
        // distance = sqrt( diff^T (R0 + R1) diff )
        double squared_norm = diff.transpose() * (R0_ + R1_) * diff;
        return std::sqrt(squared_norm);
    }
};

// d(f,g) = sqrt((f-g)^T (R0 + R1) (f-g)) / ( sqrt(f^T (R0 + R1) f) + sqrt(g^T (R0 + R1) g) )
struct SobolevPolicyNormalized {
    Eigen::MatrixXd R0_;
    Eigen::MatrixXd R1_;

    SobolevPolicyNormalized(const Eigen::MatrixXd& R0, const Eigen::MatrixXd& R1)
        : R0_(R0), R1_(R1)
    {
        if (R0_.rows() != R0_.cols()) {
            throw std::runtime_error("R0 must be a square matrix!");
        }
        if (R1_.rows() != R1_.cols()) {
            throw std::runtime_error("R1 must be square!");
        }
        if (R0_.rows() != R1_.rows()) {
            throw std::runtime_error("R0 and R1 must have the same dimension!");
        }
    }

    template <typename T1, typename T2>
    double operator()(const Eigen::MatrixBase<T1>& f,
                      const Eigen::MatrixBase<T2>& g) const 
    {
        Eigen::VectorXd diff = f - g;
        Eigen::MatrixXd M = R0_ + R1_;

        double squared_norm = diff.transpose() * M * diff;

        Eigen::VectorXd f_vector = f;
        Eigen::VectorXd g_vector = g;
        double f_squared = f_vector.transpose() * M * f_vector;
        double g_squared = g_vector.transpose() * M * g_vector;

        double denom = std::sqrt(f_squared) + std::sqrt(g_squared);
        if (denom < 1e-14) {
            return squared_norm;
        }
        return std::sqrt(squared_norm) / denom;
    }
};

// d(f,g) = sqrt((f-g)^T R0 (f-g))
class L2PolicyPartialObservability
{
private:
    Eigen::MatrixXd R0_;
public:
    // Constructor: R0 is expected to be a square matrix.
    L2PolicyPartialObservability(const Eigen::MatrixXd &R0) : R0_(R0) {}
    template <typename T1, typename T2>
    double operator()(const T1 &f, const T2 &g) const 
    {
        // Collect indices where neither f nor g are NaN
        std::vector<int> valid_indices;
        valid_indices.reserve(f.size());
        for (int j = 0; j < f.size(); ++j)
        {
            if (!std::isnan(f[j]) && !std::isnan(g[j]))
            {
                valid_indices.push_back(j);
            }
        }
        if (valid_indices.empty()) {
            // No valid indices, we can not compare functions
            return 0.0;
        }
        // Where is possible, build the vector containing the differences
        Eigen::VectorXd diff_valid(valid_indices.size());
        for (std::size_t i = 0; i < valid_indices.size(); ++i)
        {
            diff_valid(i) = f[valid_indices[i]] - g[valid_indices[i]];
        }
        // Check whether data is contiguous (as in common domain case)
        bool is_contiguous = true;
        for (std::size_t i = 1; i < valid_indices.size(); ++i)
        {
            if (valid_indices[i] != valid_indices[i - 1] + 1)
            {
                is_contiguous = false;
                break;
            }
        }
        double squared_norm = 0.0;
        if (is_contiguous)
        {
            // If contiguous, we use Eigen's block operation for imrpoved efficiency
            int start = valid_indices.front();
            Eigen::MatrixXd R0_valid = R0_.block(start, start, valid_indices.size(), valid_indices.size());
            squared_norm = diff_valid.transpose() * R0_valid * diff_valid;
        }
        else // We compute R0 manually
        {
            Eigen::MatrixXd R0_valid(valid_indices.size(), valid_indices.size());
            for (std::size_t i = 0; i < valid_indices.size(); ++i)
            {
                for (std::size_t j = 0; j < valid_indices.size(); ++j)
                {
                    R0_valid(i, j) = R0_(valid_indices[i], valid_indices[j]);
                }
            }
            squared_norm = diff_valid.transpose() * R0_valid * diff_valid;
        }
        return std::sqrt(squared_norm);
    }
};


#endif // DISSIMILARITIES_H