/*
    This file is part of ConstrainedMiniball.

    ConstrainedMiniball: Smallest Enclosing Ball with Linear Constraints.
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
#include <random>
#include <type_traits>
#if __has_include(<mpfr.h>)
    #include <unsupported/Eigen/MPRealSupport>
#endif

namespace cmb {
    using std::tuple, std::vector;

    template <class Real_t>
    using RealVector = Eigen::Matrix<Real_t, Eigen::Dynamic, 1>;

    template <class Real_t>
    using RealMatrix = Eigen::Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    template <class Derived, class Real_t>
    concept RealMatrixXpr = requires { typename Eigen::MatrixBase<Derived>; } &&
        std::is_same<typename Derived::Scalar, Real_t>::value;
    
    template <class Derived, class Real_t>
    concept RealVectorXpr = RealMatrixXpr<Derived, Real_t> && Derived::ColsAtCompileTime == 1;

    template <class Derived>
    concept FloatMatrixXpr = RealMatrixXpr<Derived, float> || RealMatrixXpr<Derived, double>;

    template <class Derived>
    concept FloatVectorXpr = RealVectorXpr<Derived, float> || RealVectorXpr<Derived, double>;

    template <class Real_t>
    class ConstrainedMiniballHelper {
        int num_points, num_linear_constraints, rank, dim;
        RealMatrix<Real_t> M;
        RealVector<Real_t> p0, v;

        public:

        template <RealMatrixXpr<Real_t> A_t, RealVectorXpr<Real_t> b_t>
        ConstrainedMiniballHelper(int dimension, const Eigen::MatrixBase<A_t>& A, const Eigen::MatrixBase<b_t>& b) : 
        num_points(0), 
        num_linear_constraints(A.rows()), 
        dim(dimension), 
        p0(RealVector<Real_t>::Zero(dim)) {
            assert(A.cols() == dim);
            assert(A.rows() == b.rows());
            M = A.eval();
            v = b.eval();
            if (M.rows() > 0) {
                rank = dim - M.completeOrthogonalDecomposition().rank();
            }
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
        auto [centre, sqRadius, success] = _constrained_miniball(X, helper);
        if ((p - centre).squaredNorm() > sqRadius) {
            helper.add_point(p);
            auto t = _constrained_miniball(X, helper);
            helper.remove_last_point();
            X.push_back(p);
            return t;
        }
        else {
            X.push_back(p);
            return tuple{centre, sqRadius, success};
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
    X, A, and b must have the same scalar type Real_t, which must be a standard floating-point type.

    RETURNS: 
    std::tuple with the following elements (in order):
    -   a column vector with Real_t entries that is the centre of the sphere of minimum radius 
        bounding every point in X. 
    -   the squared radius of the bounding sphere as a Real_t scalar.
    -   a boolean flag that is true if the solution is known to be correct to within machine precision.

    */
    template <FloatMatrixXpr X_t, FloatMatrixXpr A_t, FloatVectorXpr b_t>
    tuple<RealVector<typename X_t::Scalar>, typename X_t::Scalar, bool> constrained_miniball(
    const int d,
    const Eigen::MatrixBase<X_t>& X,
    const Eigen::MatrixBase<A_t>& A,
    const Eigen::MatrixBase<b_t>& b) {

        assert(A.rows() == b.rows());
        assert(d == X.rows());
        assert(d == A.cols());

        typedef X_t::Scalar Float_t;
        #if __has_include(<mpfr.h>)
            typedef mpfr::mpreal Real_t;
            mpfr::mpreal::set_default_prec(mpfr::digits2bits(25));
        #else
            typedef Float_t Real_t;
        #endif

        vector<RealVector<Real_t>> X_vec;
        for(auto i = 0; i < X.cols(); i++) {
            X_vec.push_back(X.col(i).cast<Real_t>());
        }
        std::random_device rd;
        std::shuffle(X_vec.begin(), X_vec.end(), rd);

        ConstrainedMiniballHelper<Real_t> helper(d, A.cast<Real_t>(), b.cast<Real_t>());

        auto [centre, sqRadius, success] = _constrained_miniball(X_vec, helper);
        RealVector<Float_t> centre_f = centre.cast<Float_t>();
        Float_t sqRadius_f = static_cast<Float_t>(sqRadius);
        if (success) {
            for(size_t i = 0; i < X.cols(); i++) {
                success &= (centre_f - X.col(i)).squaredNorm() <= sqRadius_f;
            }
            success &= (A * centre_f).isApprox(b);
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
    template <FloatMatrixXpr X_t>
    tuple<RealVector<typename X_t::Scalar>, typename X_t::Scalar, bool> miniball(
    const int d,
    const Eigen::MatrixBase<X_t>& X) {
        typedef X_t::Scalar Float_t;
        return constrained_miniball(
            d, X, 
            RealMatrix<Float_t>(0, d), 
            RealVector<Float_t>(0));
    }
}

#endif