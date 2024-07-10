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
	ConstrainedMiniball is released under the GNU Lesser General Public License
	("LGPL").

	GNU Lesser General Public License ("LGPL") copyright permissions statement:
	**************************************************************************
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Lesser General Public License as published by
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
#define NDEBUG

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <cassert>

namespace cmb {
using std::tuple, std::vector, Eigen::MatrixBase, Eigen::Matrix, Eigen::Index;

template <class Real_t> using RealVector = Matrix<Real_t, Eigen::Dynamic, 1>;

template <class Real_t>
using RealMatrix = Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;

template <class Derived>
concept MatrixXpr = requires { typename MatrixBase<Derived>; };

template <class Derived>
concept VectorXpr = requires { typename MatrixBase<Derived>; } &&
					Derived::ColsAtCompileTime == 1;

template <class Derived, class Real_t>
concept RealMatrixXpr =
	MatrixXpr<Derived> && std::same_as<typename Derived::Scalar, Real_t>;

template <class Derived, class Real_t>
concept RealVectorXpr =
	VectorXpr<Derived> && std::same_as<typename Derived::Scalar, Real_t>;

template <class Real_t> class ConstrainedMiniballHelper {
	int num_points, num_linear_constraints, dim;
	RealMatrix<Real_t> M;
	RealVector<Real_t> p0, v;
	Real_t tol;

public:
	// initialise the helper with the affine constraint Ax = b
	template <RealMatrixXpr<Real_t> A_t, RealVectorXpr<Real_t> b_t>
	ConstrainedMiniballHelper(int dimension, const MatrixBase<A_t> &A,
							  const MatrixBase<b_t> &b, Real_t tol)
		: num_points(0), num_linear_constraints(A.rows()), dim(dimension),
		  p0(RealVector<Real_t>::Zero(dim)), tol(tol) {
		assert(A.cols() == dim);
		assert(A.rows() == b.rows());
		M = A.eval();
		v = b.eval();
	}

	/* Add a constraint to the helper corresponding to
	requiring that the bounding ball pass through the point p. */
	template <RealVectorXpr<Real_t> T> void add_point(T &p) {
		if (num_points == 0) {
			// if there are no other points set so far,
			// then we just remember the point p
			assert(p.rows() == dim);
			p0 = p;
		} else {
			// Otherwise we add a new row to M and b, where Mx = b is our
			// system. We should add the row p - p0 to M and the entry 0.5 * ||p
			// - p0||^2 to b, but if we assume that p0 has been translated to
			// the origin then we can just add the row p to M and the entry 0.5
			// * ||p||^2 to b. We actually do not need to add the last entry to
			// b yet, since we can compute it from M when we actually need to
			// solve the system.
			M.conservativeResize(M.rows() + 1, Eigen::NoChange);
			M(Eigen::last, Eigen::all) = (p - p0).transpose().eval();
		}
		num_points += 1;
	}

	// remove the last point constraint that has been added to the system
	// if there is only one point so far, just set it to 0
	void remove_last_point() {
		if (num_points > 1) {
			M.conservativeResize(M.rows() - 1, Eigen::NoChange);
			num_points -= 1;
		} else if (num_points == 1) {
			num_points = 0;
			p0 *= static_cast<Real_t>(0);
		}
	}

	int subspace_rank() const noexcept {
		return dim - M.completeOrthogonalDecomposition().rank();
		;
	}

	tuple<RealVector<Real_t>, bool> solve() const {
		if (num_linear_constraints == 0 && num_points <= 1) {
			// if there are no linear constraints and at most one point,
			// then we just return the ball of radius zero passing through
			// the point if it exists, or the ball of radius zero at the origin
			// if the point does not exist.
			// Note that the program logic guarantees that p0 = 0
			// if num_points == 0 so the following is valid
			return tuple{p0, true};
		} else {
			// Otherwise we need to solve the system Mx = b.
			// We need to compute the vector b, since we did not
			// compute the entries when adding the points.
			// We also need to account for the fact that we have
			// translated everything so that p0 is origin.
			// note that if num_points == 0 then v - M*p0 has 0 rows and is
			// still valid
			RealVector<Real_t> b(M.rows());
			b << v - M.topRows(num_linear_constraints) * p0,
				0.5 * M.bottomRows(M.rows() - num_linear_constraints)
						  .rowwise()
						  .squaredNorm();
			RealVector<Real_t> c =
				M.completeOrthogonalDecomposition().pseudoInverse() * v;
			return tuple{(c + p0).eval(), (M * c - v).isZero(tol)};
		}
	}
};

/* Compute the ball of minimum radius that bounds the points in X_idx
 * and contains the points of Y_idx on its boundary, while respecting
 * the affine constraints present in helper */
template <class Real_t, RealMatrixXpr<Real_t> T>
tuple<RealVector<Real_t>, Real_t, bool>
_constrained_miniball(const MatrixBase<T> &points, vector<Index> &X_idx,
					  vector<Index> &Y_idx,
					  ConstrainedMiniballHelper<Real_t> &helper) {
	if (X_idx.size() == 0 || helper.subspace_rank() == 0) {
		// if there are no points to bound or if the constraints determine a
		// unique point, then compute the point of minimum norm
		// that satisfies the constraints
		auto [centre, success] = helper.solve();
		if (Y_idx.size() == 0) {
			// if there are no boundary points to check then we are done
			return tuple{centre, static_cast<Real_t>(0), success};
		} else {
			// get the squared radius of the ball that passes through the points
			// in Y_idx and has centre at the computed point
			// We take a maximum distance from the centre to any point in Y_idx
			// to deal with floating point inaccuracies
			Real_t sqRadius = (points(Eigen::all, Y_idx).colwise() - centre)
								  .colwise()
								  .squaredNorm()
								  .maxCoeff();
			return tuple{centre, sqRadius, success};
		}
	}
	// find the constrained miniball of all except the last point
	Index i = X_idx.back();
	X_idx.pop_back();
	auto [centre, sqRadius, success] =
		_constrained_miniball(points, X_idx, Y_idx, helper);
	if ((points.col(i) - centre).squaredNorm() > sqRadius) {
		// if the last point does not lie in the computed bounding ball,
		// add it to the list of points that will lie on the boundary of the
		// eventual ball. This determines a new constraint.
		helper.add_point(points.col(i));
		Y_idx.push_back(i);
		// compute a bounding ball with the new constraint
		auto t = _constrained_miniball(points, X_idx, Y_idx, helper);
		// undo the addition of the last point
		// this matters in nested calls to this function
		// because we assume that the function does not mutate its arguments
		helper.remove_last_point();
		Y_idx.pop_back();
		X_idx.push_back(i);
		// return the computed bounding ball
		return t;
	} else {
		// if the last point lies in the computed bounding ball, then
		// return the same ball
		// making sure to undo the removal of the last point from X_idx
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
X, A, and b must have the same scalar type Scalar.

RETURNS:
std::tuple with the following elements (in order):
-   a column vector with Scalar entries that is the centre of the sphere of
minimum radius bounding every point in X.
-   the squared radius of the bounding sphere as a Scalar scalar.
-   a boolean flag that is true if the solution is known to be correct to within
machine precision.

REMARK:
The result returned by this function defines a sphere that is guaranteed to
bound all points in the input set. Due to the limits of floating-point
computation, it is not theoretically guaranteed that this is the smallest sphere
possible. In practice the error in the radius and coordinates of the centre are
on the order of magnitude of 1e-5 for float, 1e-12 for double, and 1e-15 for
long double.

*/
template <typename Scalar, RealMatrixXpr<Scalar> X_t, RealMatrixXpr<Scalar> A_t,
		  RealVectorXpr<Scalar> b_t>
tuple<RealVector<Scalar>, Scalar, bool>
constrained_miniball(const MatrixBase<X_t> &X, const MatrixBase<A_t> &A,
					 const MatrixBase<b_t> &b) {
	assert(A.rows() == b.rows());
	assert(A.cols() == X.rows());
	int d = X.rows();
	Scalar tol = Eigen::NumTraits<Scalar>::dummy_precision();
	ConstrainedMiniballHelper<Scalar> helper(d, A, b, tol);
	RealVector<Scalar> centre(d);
	Scalar sqRadius;
	bool success;

	if (helper.subspace_rank() == 0) {
		std::tie(centre, success) = helper.solve();
	} else {
		vector<Index> X_idx(X.cols()), Y_idx;
		std::random_device rd;
		std::iota(X_idx.begin(), X_idx.end(), static_cast<Index>(0));
		std::shuffle(X_idx.begin(), X_idx.end(), rd);
		std::tie(centre, sqRadius, success) =
			_constrained_miniball(X, X_idx, Y_idx, helper);
	}
	sqRadius = (X.colwise() - centre).colwise().squaredNorm().maxCoeff();

	return tuple{centre, sqRadius, success};
}

/* MINIBALL ALGORITHM
Returns the sphere of minimum radius that bounds all points in X.

INPUTS:
-   d is the dimension of the ambient space.
-   X is a vector of points in R^d.
We refer to the scalar type of X as Real_t, which must be a standard
floating-point type.

RETURNS:
std::tuple with the following elements (in order):
-   a column vector with Real_t entries that is the centre of the sphere of
minimum radius bounding every point in X.
-   the squared radius of the bounding sphere as a Real_t scalar.
-   a boolean flag that is true if the solution is known to be correct to within
machine precision.
*/
template <typename Scalar, RealMatrixXpr<Scalar> X_t>
tuple<RealVector<Scalar>, Scalar, bool> miniball(const MatrixBase<X_t> &X) {
	return constrained_miniball<Scalar>(X, RealMatrix<Scalar>(0, X.rows()),
										RealVector<Scalar>(0));
}
} // namespace cmb
#endif
