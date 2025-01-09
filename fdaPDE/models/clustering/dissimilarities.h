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
    double operator()(const T1& f,
                      const T2& g) const 
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
        double val_fg = diff.transpose() * R0_ * diff;
        double ff = f.transpose() * R0_ * f;
        double gg = g.transpose() * R0_ * g;
        double denom = std::sqrt(ff) + std::sqrt(gg);

        // If denom is extremely small, handle gracefully (avoid /0).
        if (denom < 1e-14) {
            return 0.0; // or consider returning sqrt(val_fg)
        }
        return std::sqrt(val_fg) / denom;
    }
};


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
    double operator()(const Eigen::MatrixBase<T1>& a,
                      const Eigen::MatrixBase<T2>& b) const 
    {
        Eigen::VectorXd diff = a - b;
        double val = diff.transpose() * R1_ * diff;
        return std::sqrt(val);
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
    double operator()(const Eigen::MatrixBase<T1>& a,
                      const Eigen::MatrixBase<T2>& b) const
    {
        Eigen::VectorXd diff = a - b;
        double val_ab = diff.transpose() * R1_ * diff;
        double aa = a.transpose() * R1_ * a;
        double bb = b.transpose() * R1_ * b;
        double denom = std::sqrt(aa) + std::sqrt(bb);

        if (denom < 1e-14) {
            return 0.0; 
        }
        return std::sqrt(val_ab) / denom;
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
        double val = diff.transpose() * (R0_ + R1_) * diff;
        return std::sqrt(val);
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

        double val_fg = diff.transpose() * M * diff;

        Eigen::VectorXd f_vector = f;
        Eigen::VectorXd g_vector = g;
        double val_ff = f_vector.transpose() * M * f_vector;
        double val_gg = g_vector.transpose() * M * g_vector;

        double denom = std::sqrt(val_ff) + std::sqrt(val_gg);
        if (denom < 1e-14) {
            return 0.0;
        }
        return std::sqrt(val_fg) / denom;
    }
};

#endif // DISSIMILARITIES_H