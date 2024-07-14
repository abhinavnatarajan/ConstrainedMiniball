/*
        This file is part of ConstrainedMiniball.

        ConstrainedMiniball: Smallest Enclosing Ball with Affine Constraints.
        Based on: E. Welzl, “Smallest enclosing disks (balls and ellipsoids),”
        in New Results and New Trends in Computer Science, H. Maurer, Ed.,
        in Lecture Notes in Computer Science. Berlin, Heidelberg: Springer,
        1991, pp. 359–370. doi: 10.1007/BFb0038202.

        Project homepage: http://github.com/abhinavnatarajan/ConstrainedMiniball

        Copyright (c) 2023 Abhinav Natarajan

        Contributors:
        Abhinav Natarajan

        Licensing:
        ConstrainedMiniball is released under the GNU General Public
   License
        ("GPL").

        GNU Lesser General Public License ("GPL") copyright permissions
   statement:
        **************************************************************************
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published
   by the Free Software Foundation, either version 3 of the License, or (at your
   option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <CGAL/Gmpzf.h>
#include <CGAL/QP_functions.h>
#include <CGAL/QP_models.h>

#include <Eigen/Dense>

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

namespace cmb {
using SolverScalarType   = CGAL::Gmpzf;                       // exact floats
using SolutionScalarType = CGAL::Quotient<SolverScalarType>;  // exact rational numbers

namespace detail {

using std::tuple, std::max, std::vector, Eigen::MatrixBase, Eigen::Matrix, Eigen::Vector,
	Eigen::MatrixXd, Eigen::VectorXd, Eigen::Index, std::same_as;

template <class Real_t> using RealVector = Matrix<Real_t, Eigen::Dynamic, 1>;

template <class Real_t> using RealMatrix = Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;

template <class Derived>
concept MatrixExpr = requires { typename MatrixBase<Derived>; };

template <class Derived>
concept VectorExpr = requires { typename MatrixBase<Derived>; } && Derived::ColsAtCompileTime == 1;

template <class Derived, class Real_t>
concept RealMatrixExpr = MatrixExpr<Derived> && same_as<typename Derived::Scalar, Real_t>;

template <class Derived, class Real_t>
concept RealVectorExpr = VectorExpr<Derived> && same_as<typename Derived::Scalar, Real_t>;

using QuadraticProgram         = CGAL::Quadratic_program<SolverScalarType>;
using QuadraticProgramSolution = CGAL::Quadratic_program_solution<SolverScalarType>;

class ConstrainedMiniballSolver {
	const RealMatrix<SolverScalarType> A, points;
	const RealVector<SolverScalarType> b;
	RealMatrix<SolverScalarType>       lhs;
	RealVector<SolverScalarType>       rhs;
	vector<Index>                      boundary_points;
	static constexpr double            tol = Eigen::NumTraits<double>::dummy_precision();

	/* Add a constraint to the helper corresponding to
	requiring that the bounding ball pass through the point p. */
	void add_point(Index& i) {
		boundary_points.push_back(i);
	}

	// remove the last point constraint that has been added to the system
	// if there is only one point so far, just set it to 0
	void remove_last_point() {
		boundary_points.pop_back();
	}

	// return the dimension of the affine subspace defined by the constraints
	// TODO: this might not work if the constraints are not linearly independent
	int subspace_rank() const {
		return A.cols() - (A.rows() + boundary_points.size() - 1);
	}

	void setup_equations() {
		int num_linear_constraints = A.rows();
		int num_point_constraints  = max(static_cast<int>(boundary_points.size()) - 1, 0);
		int total_num_constraints  = num_linear_constraints + num_point_constraints;
		assert(total_num_constraints > 0 && "Need at least one constraint");
		int dim = points.rows();
		lhs.conservativeResize(total_num_constraints, dim);
		rhs.conservativeResize(total_num_constraints, Eigen::NoChange);
		lhs.topRows(A.rows()) = A;
		if (boundary_points.size() == 0) {
			rhs = b;
		} else {
			rhs.topRows(A.rows()) = b - A * points(Eigen::all, boundary_points[0]);
			if (num_point_constraints > 0) {
				auto temp = points(Eigen::all, boundary_points).transpose();
				lhs.bottomRows(num_point_constraints) =
					temp.bottomRows(num_point_constraints).rowwise() - temp.row(0);
				rhs.bottomRows(num_point_constraints) =
					0.5 * lhs.bottomRows(num_point_constraints).rowwise().squaredNorm();
			}
		}
	}

	tuple<RealVector<SolutionScalarType>, SolutionScalarType, bool> solve_intermediate() {
		RealVector<SolutionScalarType> p0(points.rows());
		if (boundary_points.size() == 0) {
			p0 = RealVector<SolutionScalarType>::Zero(points.rows());
		} else {
			p0 = points(Eigen::all, boundary_points[0])
			         .template cast<SolutionScalarType>();  // from SolverExactType
		}
		if (A.rows() == 0 && boundary_points.size() <= 1) {
			return tuple{p0, static_cast<SolutionScalarType>(0.0), true};
		} else {
			setup_equations();
			QuadraticProgram qp(CGAL::EQUAL,
			                    false,
			                    SolverScalarType(0),
			                    false,
			                    SolverScalarType(0));
			for (int i = 0; i < lhs.rows(); i++) {
				qp.set_b(i, rhs(i));
				for (int j = 0; j < lhs.cols(); j++) {
					// intentional transpose
					// see CGAL API
					// https://doc.cgal.org/latest/QP_solver/classCGAL_1_1Quadratic__program.html
					qp.set_a(j, i, lhs(i, j));
				}
			}
			for (int j = 0; j < lhs.cols(); j++) {
				qp.set_d(j, j, 2);
			}
			QuadraticProgramSolution soln = CGAL::solve_quadratic_program(qp, SolverScalarType());
			bool success = soln.solves_quadratic_program(qp) && !soln.is_infeasible();
			assert(success && "QP solver failed");
			SolutionScalarType sqRadius = 0.0;
			if (boundary_points.size() > 0) {
				sqRadius = soln.objective_value();
			}
			RealVector<SolutionScalarType> c(points.rows());
			for (auto [i, j] = tuple{soln.variable_values_begin(), c.begin()};
			     i != soln.variable_values_end();
			     i++, j++) {
				*j = *i;
			}
			return tuple{(c + p0).eval(), sqRadius, success};
		}
	}

  public:
	// initialise the helper with the affine constraint Ax = b
	// dimension explicitly passed in because A and b can be empty
	template <RealMatrixExpr<SolverScalarType> points_t,
	          RealMatrixExpr<SolverScalarType> A_t,
	          RealVectorExpr<SolverScalarType> b_t>
	ConstrainedMiniballSolver(const points_t& points, const A_t& A, const b_t& b) :
		points(points.eval()),
		A(A.eval()),
		b(b.eval()) {
		assert(A.cols() == points.rows() && "A.cols() != points.rows()");
		assert(A.rows() == b.rows() && "A.rows() != b.rows()");
	}

	/* Compute the ball of minimum radius that bounds the points in X_idx
	 * and contains the points of Y_idx on its boundary, while respecting
	 * the affine constraints present in helper */
	tuple<RealVector<SolutionScalarType>, SolutionScalarType, bool> solve(vector<Index>& X_idx) {
		if (X_idx.size() == 0 || subspace_rank() == 0) {
			// if there are no points to bound or if the constraints determine a
			// unique point, then compute the point of minimum norm
			// that satisfies the constraints
			return solve_intermediate();
		}
		// find the constrained miniball of all except the last point
		Index i = X_idx.back();
		X_idx.pop_back();
		auto [centre, sqRadius, success] = solve(X_idx);
		auto sqDistance =
			(points.col(i).template cast<SolutionScalarType>() - centre).squaredNorm();
		if (sqDistance > sqRadius) {
			// if the last point does not lie in the computed bounding ball,
			// add it to the list of points that will lie on the boundary of the
			// eventual ball. This determines a new constraint.
			add_point(i);
			// compute a bounding ball with the new constraint
			std::tie(centre, sqRadius, success) = solve(X_idx);
			// undo the addition of the last point
			// this matters in nested calls to this function
			// because we assume that the function does not mutate its arguments
			remove_last_point();
		}
		X_idx.push_back(i);
		return tuple{centre, sqRadius, success};
	}
};
}  // namespace detail

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
template <detail::MatrixExpr X_t, detail::MatrixExpr A_t, detail::VectorExpr b_t>
	requires std::same_as<typename X_t::Scalar, typename A_t::Scalar> &&
             std::same_as<typename A_t::Scalar, typename b_t::Scalar>
std::tuple<detail::RealVector<SolutionScalarType>, SolutionScalarType, bool>
constrained_miniball(const X_t& X, const A_t& A, const b_t& b) {
	using namespace detail;
	using Real_t = X_t::Scalar;
	assert(A.rows() == b.rows() && "A.rows() != b.rows()");
	assert(A.cols() == X.rows() && "A.cols() != X.rows()");
	vector<Index> X_idx(X.cols());
	std::iota(X_idx.begin(), X_idx.end(), static_cast<Index>(0));
	// shuffle the points
	std::random_device rd;
	std::shuffle(X_idx.begin(), X_idx.end(), rd);

	// Get the result
	ConstrainedMiniballSolver solver(X.template cast<SolverScalarType>(),
	                                 A.template cast<SolverScalarType>(),
	                                 b.template cast<SolverScalarType>());
	return solver.solve(X_idx);
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
template <detail::MatrixExpr X_t>
std::tuple<detail::RealVector<SolutionScalarType>, SolutionScalarType, bool>
miniball(const X_t& X) {
	using namespace detail;
	using Real_t = X_t::Scalar;
	using Mat    = Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;
	using Vec    = Vector<Real_t, Eigen::Dynamic>;
	return constrained_miniball(X, Mat(0, X.rows()), Vec(0));
}

template <detail::MatrixExpr T>
std::tuple<detail::RealMatrix<typename T::Scalar>, detail::RealVector<typename T::Scalar>>
equidistant_subspace(const T& X) {
	using namespace detail;
	using Real_t         = T::Scalar;
	int                n = X.cols();
	RealMatrix<Real_t> E(n - 1, X.rows());
	RealVector<Real_t> b(n - 1);
	if (n > 1) {
		b = static_cast<Real_t>(0.5) *
		    (X.rightCols(n - 1).colwise().squaredNorm().array() - X.col(0).squaredNorm())
		        .transpose();
		E = (X.rightCols(n - 1).colwise() - X.col(0)).transpose();
	}
	return tuple{E, b};
}

}  // namespace cmb
