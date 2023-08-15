/* 
Copyright 2023 Abhinav Natarajan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
#pragma once
#ifndef CONSTRAINED_MINIBALL_H
#define CONSTRAINED_MINIBALL_H

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <type_traits>

namespace cmb {
    using std::tuple, std::vector;

    template <class Real_t>
    using RealVector = Eigen::Matrix<Real_t, Eigen::Dynamic, 1>;

    template <class Real_t>
    class ConstrainedMiniballHelper {
        int num_points, num_linear_constraints, rank, dim;
        Eigen::Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic> M;
        RealVector<Real_t> p0, v;

        public:

        template <class DA, class Db>
        ConstrainedMiniballHelper(int dimension, const Eigen::MatrixBase<DA>& A, const Eigen::MatrixBase<Db>& b) : 
        num_points(0), 
        num_linear_constraints(A.rows()), 
        dim(dimension), 
        p0(RealVector<Real_t>::Zero(dim)) {
            #ifndef NDEBUG
                assert(A.cols() == dim);
                assert(A.rows() == b.rows());
            #endif
            M = A.eval();
            v = b.eval();
            if (M.rows() > 0) {
                rank = dim - M.completeOrthogonalDecomposition().rank();
            }
        }

        void add_point(RealVector<Real_t>& p) {
            if (num_points == 0) {
                #ifndef NDEBUG
                    assert(p.rows() == dim);
                #endif
                p0 = p;
            }
            else {
                M.conservativeResize(M.rows()+1, Eigen::NoChange);
                M(Eigen::last, Eigen::all) = (p - p0).transpose().eval();
                rank = dim - M.completeOrthogonalDecomposition().rank();
            }
            num_points += 1;
        }

        void remove_last_point() {
            if (num_points > 1) {
                M.conservativeResize(M.rows()-1, Eigen::NoChange);
                rank = dim - M.completeOrthogonalDecomposition().rank();
                num_points -= 1;
            }
            else if (num_points == 1) {
                num_points = 0;
                p0 *= static_cast<Real_t>(0);
            }
        }
        
        int subspace_rank() const noexcept {
            return rank;
        }

        tuple<RealVector<Real_t>, Real_t, bool> solve() const {
            if (num_linear_constraints == 0 && num_points <= 1) {
                // note that the program logic guarantees that p0 = 0 if num_points == 0
                // so the following is valid
                return tuple{p0, static_cast<Real_t>(0), true};
            }
            else {
                RealVector<Real_t> rhs(M.rows());
                // note that if num_points == 0 then v - A*p0 has 0 rows and is still valid
                rhs << v - M.topRows(num_linear_constraints) * p0, 
                    0.5 * M.bottomRows(M.rows() - num_linear_constraints).rowwise().squaredNorm();
                RealVector<Real_t> c = M.completeOrthogonalDecomposition().pseudoInverse() * rhs;
                return tuple{(c + p0).eval(), static_cast<Real_t>(c.squaredNorm()), (M * c).isApprox(rhs)};
            }
        }
    };

    template <class Real_t>
    tuple<RealVector<Real_t>, Real_t, bool> _constrained_miniball(
    vector<RealVector<Real_t>>& X, 
    ConstrainedMiniballHelper<Real_t>& helper) {
        if (X.size() == 0 || helper.subspace_rank() == 0) {
            return helper.solve();
        }
        RealVector<Real_t> p = X.back();
        X.pop_back();
        auto [centre, sqRadius, flag] = _constrained_miniball(X, helper);
        if ((p - centre).squaredNorm() > sqRadius) {
            helper.add_point(p);
            auto t = _constrained_miniball(X, helper);
            helper.remove_last_point();
            X.push_back(p);
            return t;
        }
        else {
            X.push_back(p);
            return tuple{centre, sqRadius, flag};
        }
    }

    /* 
    CONSTRAINED MINIBALL ALGORITHM 
    Returns the sphere of minimum radius that bounds all points in X, 
    and whose centre lies in a given affine subspace. 

    INPUTS: 
    -   d is the dimension of the ambient space.
    -   X is a vector of points in R^d.
    -   A is a (m x d) matrix with m <= d.
    -   b is a vector in R^m such that Ax = b defines an affine subspace of R^d. 
    X, A, and b must have the same scalar type, which we refer to as Real_t.
    If Real_t is not a standard floating-point type, Eigen support for the type must be added.
    See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html for details.

    RETURNS: 
    std::tuple with the following elements (in order):
    -   a column vector with Real_t entries that is the centre of the sphere of minimum radius 
        bounding every point in X. 
    -   the squared radius of the bounding sphere as a Real_t scalar.
    -   a boolean flag that is true if the solution is known to be correct to within machine precision.

    */
    template <class DX, class DA, class Db>
    tuple<RealVector<typename DX::Scalar>, typename DX::Scalar, bool> constrained_miniball(
    const int d,
    const Eigen::MatrixBase<DX>& X,
    const Eigen::MatrixBase<DA>& A,
    const Eigen::MatrixBase<Db>& b) {
        #ifndef NDEBUG
            static_assert(Db::ColsAtCompileTime == 1, "b must be a column vector");
            static_assert(std::is_same<DA::Scalar, DX::Scalar>::value, "A and X must have the same type of scalars.");
            static_assert(std::is_same<Db::Scalar, DX::Scalar>::value, "b and X must have the same type of scalars.");
            assert(A.rows() == b.rows());
            assert(d == X.rows());
            assert(d == A.cols());
        #endif
        vector<RealVector<typename DX::Scalar>> X_vec;
        for(auto i = 0; i < X.cols(); i++) {
            X_vec.push_back(X.col(i));
        }
        std::random_device rd;
        std::shuffle(X_vec.begin(), X_vec.end(), rd);
        ConstrainedMiniballHelper<typename DX::Scalar> helper(d, A, b);
        auto [centre, sqRadius, flag] = _constrained_miniball(X_vec, helper);
        if (flag) {
            for(size_t i = 0; i < X.cols(); i++) {
                flag &= (centre - X.col(i)).squaredNorm() <= sqRadius;
            }
            flag &= (A * centre).isApprox(b);
        }
        return tuple{centre, sqRadius, flag};
    }

    /* MINIBALL ALGORITHM 
    Returns the sphere of minimum radius that bounds all points in X. 

    INPUTS: 
    -   d is the dimension of the ambient space.
    -   X is a vector of points in R^d.
    We refer to the scalar type of X as Real_t.
    If Real_t is not a standard floating-point type, Eigen support for the type must be added.
    See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html for details.

    RETURNS: 
    std::tuple with the following elements (in order):
    -   a column vector with Real_t entries that is the centre of the sphere of minimum radius 
        bounding every point in X. 
    -   the squared radius of the bounding sphere as a Real_t scalar.
    -   a boolean flag that is true if the solution is known to be correct to within machine precision.

    */
    template <class DX>
    tuple<RealVector<typename DX::Scalar>, typename DX::Scalar, bool> miniball(
    const int d,
    const Eigen::MatrixBase<DX>& X) {
        typedef DX::Scalar Real_t;
        return constrained_miniball(
            d, X, 
            Eigen::Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>(0, d), 
            RealVector<Real_t>(0));
    }
}

#endif