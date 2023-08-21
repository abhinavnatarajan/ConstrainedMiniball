/*
    This file is part of ConstrainedMiniball.

    ConstrainedMiniball: Smallest Enclosing Ball with Affine Constraints.
    Based on: E. Welzl, “Smallest enclosing disks (balls and ellipsoids),” 
    in New Results and New Trends in Computer Science, H. Maurer, Ed., 
    in Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 
    1991, pp. 359–370. doi: 10.1007/BFb0038202.

    Project homepage:    http://github.com/abhinavnatarajan/ConstrainedMiniball

    Copyright (c) 2023 Abhinav Natarajan

    Contributors:
    Abhinav Natarajan

    Licensing:
    ConstrainedMiniball is released under the GNU Lesser General Public License ("LGPL").

    GNU Lesser General Public License ("LGPL") copyright permissions statement:
    **************************************************************************
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef CONSTRAINED_MINIBALL_H
#define CONSTRAINED_MINIBALL_H

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>
#if __has_include(<mpfr.h>) && !defined CMB_NO_MPFR
    #if defined __MPFR_H && !defined MPFR_USE_NO_MACRO
        #error "This header uses the mpreal C++ library, which relies on <mpfr.h> being included after MPFR_USE_NO_MACRO has been defined. It seems that your program already includes <mpfr.h> elsewhere without first defining MPFR_USE_NO_MACRO. To solve this problem, include this header earlier in your code."
    #else
        #define CMB_USE_MPFR
        #include <unsupported/Eigen/MPRealSupport>
    #endif
#endif
#ifndef NDEBUG
    #include <iostream>
    #include <string>
    #include <sstream>
    template <typename T>
    std::string to_string_with_precision(const T a_value, const int n = 18)
    {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return std::move(out).str();
    }
#endif

namespace cmb {
    using std::tuple, std::vector, Eigen::MatrixBase, Eigen::Matrix, Eigen::Index;

    template <class Real_t>
    using RealVector = Matrix<Real_t, Eigen::Dynamic, 1>;

    template <class Real_t>
    using RealMatrix = Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;

    template <class Derived> 
    concept MatrixXpr = requires { typename MatrixBase<Derived>; };

    template <class Derived> 
    concept VectorXpr = requires { typename MatrixBase<Derived>; } && Derived::ColsAtCompileTime == 1;
    
    template <class Derived, class Real_t>
    concept RealMatrixXpr = MatrixXpr<Derived> && std::same_as<typename Derived::Scalar, Real_t>;
    
    template <class Derived, class Real_t>
    concept RealVectorXpr = VectorXpr<Derived> && std::same_as<typename Derived::Scalar, Real_t>;

    template <class T>
    concept FloatType = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, long double>;

    template <class Real_t>
    class ConstrainedMiniballHelper {
        int num_points, num_linear_constraints, dim;
        RealMatrix<Real_t> M;
        RealVector<Real_t> p0, v;
        Real_t tol;

        public:

        template <RealMatrixXpr<Real_t> A_t, RealVectorXpr<Real_t> b_t>
        ConstrainedMiniballHelper(
            int dimension, const MatrixBase<A_t>& A, const MatrixBase<b_t>& b, Real_t tol) : 
            num_points(0), 
            num_linear_constraints(A.rows()), 
            dim(dimension), 
            p0(RealVector<Real_t>::Zero(dim)),
            tol(tol)
            {
                assert(A.cols() == dim);
                assert(A.rows() == b.rows());
                M = A.eval();
                v = b.eval();
        }

        template <RealVectorXpr<Real_t> T>
        void add_point(T& p) {
            if (num_points == 0) {
                assert(p.rows() == dim);
                p0 = p;
            }
            else {
                M.conservativeResize(M.rows()+1, Eigen::NoChange);
                M(Eigen::last, Eigen::all) = (p - p0).transpose().eval();
            }
            num_points += 1;
        }

        void remove_last_point() {
            if (num_points > 1) {
                M.conservativeResize(M.rows()-1, Eigen::NoChange);
                num_points -= 1;
            }
            else if (num_points == 1) {
                num_points = 0;
                p0 *= static_cast<Real_t>(0);
            }
        }
        
        int subspace_rank() const noexcept {
            return dim - M.completeOrthogonalDecomposition().rank();;
        }

        tuple<RealVector<Real_t>, bool> solve() const {
            if (num_linear_constraints == 0 && num_points <= 1) {
                // note that the program logic guarantees that p0 = 0 if num_points == 0
                // so the following is valid
                return tuple{p0, true};
            }
            else {
                RealVector<Real_t> rhs(M.rows());
                // note that if num_points == 0 then v - A*p0 has 0 rows and is still valid
                rhs << v - M.topRows(num_linear_constraints) * p0, 
                    0.5 * M.bottomRows(M.rows() - num_linear_constraints).rowwise().squaredNorm();
                RealVector<Real_t> c = M.completeOrthogonalDecomposition().pseudoInverse() * rhs;
                return tuple{(c + p0).eval(), (M * c - rhs).isZero(tol)};
            }
        }
    };

    template <class Real_t, RealMatrixXpr<Real_t> T>
    tuple<RealVector<Real_t>, Real_t, bool> _constrained_miniball(
    const MatrixBase<T>& points,
    vector<Index>& X_idx, 
    vector<Index>& Y_idx, 
    ConstrainedMiniballHelper<Real_t>& helper) {
        if (X_idx.size() == 0 || helper.subspace_rank() == 0) {
            auto [centre, success] = helper.solve();
            if (Y_idx.size() == 0) {
                return tuple{centre, static_cast<Real_t>(0), success};
            }
            else {
                Real_t sqRadius = (points(Eigen::all, Y_idx).colwise() - centre).colwise().squaredNorm().maxCoeff();
                return tuple{centre, sqRadius, success};
            }
        }
        Index i = X_idx.back();
        X_idx.pop_back();
        auto [centre, sqRadius, success] = _constrained_miniball(points, X_idx, Y_idx, helper);
        if ((points.col(i) - centre).squaredNorm() > sqRadius) {
            helper.add_point(points.col(i));
            Y_idx.push_back(i);
            auto t = _constrained_miniball(points, X_idx, Y_idx, helper);
            helper.remove_last_point();
            Y_idx.pop_back();
            X_idx.push_back(i);
            return t;
        }
        else {
            X_idx.push_back(i);
            return tuple{centre, sqRadius, success};
        }
    }

    /* 
    CONSTRAINED MINIBALL ALGORITHM 
    Returns the sphere of minimum radius that bounds all points in X, 
    and whose centre lies in a given affine subspace. 

    INPUTS: 
    -   d is the dimension of the ambient space.
    -   X is a matrix whose columns are points in R^d.
    -   A is a (m x d) matrix with m <= d.
    -   b is a vector in R^m such that Ax = b defines an affine subspace of R^d. 
    X, A, and b must have the same scalar type Float_t, which must be a standard floating-point type.

    RETURNS: 
    std::tuple with the following elements (in order):
    -   a column vector with Float_t entries that is the centre of the sphere of minimum radius 
        bounding every point in X. 
    -   the squared radius of the bounding sphere as a Float_t scalar.
    -   a boolean flag that is true if the solution is known to be correct to within machine precision.

    REMARK:
    The result returned by this function defines a sphere that is guaranteed to bound all points in the input set. Due to the limits of floating-point computation, it is not theoretically guaranteed that this is the smallest sphere possible. In practice the error in the radius and coordinates of the centre are on the order of magnitude of 1e-5 for float, 1e-12 for double, and 1e-15 for long double.

    */
    template <FloatType Float_t, RealMatrixXpr<Float_t> X_t, RealMatrixXpr<Float_t> A_t, RealVectorXpr<Float_t> b_t>
    tuple<RealVector<Float_t>, Float_t, bool> constrained_miniball(
    const int d,
    const MatrixBase<X_t>& X,
    const MatrixBase<A_t>& A,
    const MatrixBase<b_t>& b) {

        assert(A.rows() == b.rows());
        assert(d == X.rows());
        assert(d == A.cols());

        #ifdef CMB_USE_MPFR
            constexpr int digits_precision = std::numeric_limits<Float_t>::digits10 + 10;
            typedef mpfr::mpreal Real_t;
            mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits_precision));
        #else
            typedef Float_t Real_t;
        #endif
        Real_t tol = static_cast<Real_t>(Eigen::NumTraits<Float_t>::dummy_precision());
        ConstrainedMiniballHelper<Real_t> helper(d, A.cast<Real_t>(), b.cast<Real_t>(), tol);
        
        const RealMatrix<Real_t> points = X.cast<Real_t>();
        RealVector<Real_t> centre(d);
        Real_t sqRadius;
        RealVector<Float_t> centre_f(d);
        Float_t sqRadius_f;
        bool success;

        if (helper.subspace_rank() == 0) {
            std::tie(centre, success) = helper.solve();
            centre_f = centre.cast<Float_t>();
            sqRadius_f = (X.colwise() - centre_f).colwise().squaredNorm().maxCoeff();
        }
        else {
            vector<Index> X_idx(X.cols()), Y_idx;
            std::random_device rd;
            std::iota(X_idx.begin(), X_idx.end(), static_cast<Index>(0));
            std::shuffle(X_idx.begin(), X_idx.end(), rd);
            std::tie(centre, sqRadius, success) = _constrained_miniball(points, X_idx, Y_idx, helper);
            centre_f = centre.cast<Float_t>();
            if (std::is_same<Float_t, Real_t>::value) {
                sqRadius_f = static_cast<Float_t>(sqRadius);
            }
            else {
                sqRadius_f = (X.colwise() - centre_f).colwise().squaredNorm().maxCoeff();
            }
        }
        
        return tuple{centre_f, sqRadius_f, success};
    }

    /* MINIBALL ALGORITHM 
    Returns the sphere of minimum radius that bounds all points in X. 

    INPUTS: 
    -   d is the dimension of the ambient space.
    -   X is a vector of points in R^d.
    We refer to the scalar type of X as Real_t, which must be a standard floating-point type.

    RETURNS: 
    std::tuple with the following elements (in order):
    -   a column vector with Real_t entries that is the centre of the sphere of minimum radius 
        bounding every point in X. 
    -   the squared radius of the bounding sphere as a Real_t scalar.
    -   a boolean flag that is true if the solution is known to be correct to within machine precision.
    */
    template <FloatType Float_t, RealMatrixXpr<Float_t> X_t>
    tuple<RealVector<Float_t>, Float_t, bool> miniball(
    const int d,
    const MatrixBase<X_t>& X) {
        typedef X_t::Scalar Float_t;
        return constrained_miniball<Float_t>(
            d, X, 
            RealMatrix<Float_t>(0, d), 
            RealVector<Float_t>(0));
    }
}
#endif