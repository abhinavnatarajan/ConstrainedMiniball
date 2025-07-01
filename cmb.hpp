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
#ifndef CMB_HPP
	#define CMB_HPP

	#include <CGAL/Gmpzf.h>
	#include <CGAL/QP_functions.h>
	#include <CGAL/QP_models.h>

	#include <Eigen/Dense>

	#include <algorithm>
	#include <random>
	#include <tuple>
	#include <vector>

namespace cmb {
using SolutionExactType = CGAL::Quotient<CGAL::Gmpzf>;  // exact rational numbers

enum class SolutionPrecision : std::uint8_t {
	EXACT,  // exact rational numbers
	DOUBLE  // C++ doubles
};

template <SolutionPrecision S>
using SolutionType = std::conditional_t<S == SolutionPrecision::EXACT, SolutionExactType, double>;

namespace detail {

using CGAL::Gmpzf;  // exact floats
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

using QuadraticProgram         = CGAL::Quadratic_program<Gmpzf>;
using QuadraticProgramSolution = CGAL::Quadratic_program_solution<Gmpzf>;

class ConstrainedMiniballSolver {
	RealMatrix<Gmpzf>       m_A, m_points;  // not changed after construction
	Index                   m_rank_A_ub, m_dim_points;
	RealVector<Gmpzf>       m_b;
	RealMatrix<Gmpzf>       m_lhs;
	RealVector<Gmpzf>       m_rhs;
	vector<Index>           m_boundary_points;
	static constexpr double tol = Eigen::NumTraits<double>::dummy_precision();

	// Add a constraint to the helper corresponding to
	// requiring that the bounding ball pass through the point p.
	void add_point(Index& i) {
		m_boundary_points.push_back(i);
	}

	// Remove the last point constraint that has been added to the system.
	// If there is only one point so far, just set it to 0.
	void remove_last_point() {
		m_boundary_points.pop_back();
	}

	// Return a lower bound on the dimension of the affine subspace defined by the constraints.
	// With high probability this function returns the actual subspace rank.
	[[nodiscard]]
	auto subspace_rank_lb() const -> Index {
		// The static_cast below is safe because boundary_points never exceeds the number of points.
		return std::max(
			static_cast<Index>(0),
			m_dim_points - m_rank_A_ub -
				(static_cast<Index>(m_boundary_points.size()) - static_cast<Index>(1))
		);
	}

	void setup_equations() {
		Index num_linear_constraints = m_A.rows();
		Index num_point_constraints =
			max(static_cast<Index>(m_boundary_points.size()) - 1, static_cast<Index>(0));
		Index total_num_constraints = num_linear_constraints + num_point_constraints;
		assert(total_num_constraints > 0 && "Need at least one constraint");
		m_lhs.conservativeResize(total_num_constraints, m_dim_points);
		m_rhs.conservativeResize(total_num_constraints, Eigen::NoChange);
		m_lhs.topRows(m_A.rows()) = m_A;
		if (m_boundary_points.size() == 0) {
			m_rhs = m_b;
		} else {
			m_rhs.topRows(m_A.rows()) = m_b - m_A * m_points(Eigen::all, m_boundary_points[0]);
			if (num_point_constraints > 0) {
				// temp = X^T
				auto&& temp = m_points(Eigen::all, m_boundary_points).transpose();
				m_lhs.bottomRows(num_point_constraints) =
					// [x_i - x_0]^T_{i > 0}
					temp.bottomRows(num_point_constraints).rowwise() - temp.row(0);
				m_rhs.bottomRows(num_point_constraints) =
					// [ 1/2 * |x_i - x_0|^2 ]_{i > 0}
					0.5 * m_lhs.bottomRows(num_point_constraints).rowwise().squaredNorm();
			}
		}
	}

	// Computes the minimum circumball of the points in m_boundary_points whose centre lies
	// in the affine subspace defined by the constraints Ax = b.
	// We do not assume that the equidistance subspace of the boundary points
	// and the affine space defined by Ax = b are linearly independent, or even that A has full
	// rank.
	auto solve_intermediate() -> tuple<RealVector<SolutionExactType>, SolutionExactType, bool> {
		// We translate the entire system by x_0,
		// where x_0 is the first boundary point,
		// to simplify the equations.
		RealVector<SolutionExactType> p0(m_points.rows());
		if (m_boundary_points.size() == 0) {
			// No boundary points, no translation.
			p0 = RealVector<SolutionExactType>::Zero(m_points.rows());
		} else {
			p0 = m_points(Eigen::all, m_boundary_points[0])
			         .template cast<SolutionExactType>();  // from SolverExactType
		}

		if (m_A.rows() == 0 && m_boundary_points.size() <= 1) {
			// Only one point and no linear constraints;
			// return circle of radius 0 at the point.
			return tuple{p0, static_cast<SolutionExactType>(0.0), true};
		} else {
			setup_equations();
			QuadraticProgram qp(CGAL::EQUAL, false, Gmpzf(0), false, Gmpzf(0));
			// WARNING: COUNTER MIGHT OVERFLOW
			for (int i = 0; i < m_lhs.rows(); i++) {
				qp.set_b(i, m_rhs(i));
				// WARNING: COUNTER MIGHT OVERFLOW
				for (int j = 0; j < m_lhs.cols(); j++) {
					// intentional transpose
					// see CGAL API
					// https://doc.cgal.org/latest/QP_solver/classCGAL_1_1Quadratic__program.html
					qp.set_a(j, i, m_lhs(i, j));
				}
			}
			// WARNING: COUNTER MIGHT OVERFLOW
			for (int j = 0; j < m_lhs.cols(); j++) {
				qp.set_d(j, j, 2);
			}
			QuadraticProgramSolution soln = CGAL::solve_quadratic_program(qp, Gmpzf());
			bool success = soln.solves_quadratic_program(qp) && !soln.is_infeasible();
			assert(success && "QP solver failed");
			SolutionExactType sqRadius = 0.0;
			if (m_boundary_points.size() > 0) {
				sqRadius = soln.objective_value();
			}
			RealVector<SolutionExactType> c(m_points.rows());
			for (auto [i, j] = tuple{soln.variable_values_begin(), c.begin()};
			     i != soln.variable_values_end();
			     i++, j++) {
				*j = *i;
			}
			return tuple{(c + p0).eval(), sqRadius, success};
		}
	}

  public:
	// Initialise the helper with the affine constraint Ax = b.
	template <RealMatrixExpr<Gmpzf> points_t, RealMatrixExpr<Gmpzf> A_t, RealVectorExpr<Gmpzf> b_t>
	ConstrainedMiniballSolver(const points_t& points, const A_t& A, const b_t& b) :
		m_points(points.eval()),
		m_A(A.eval()),
		m_b(b.eval()) {
		assert(A.cols() == points.rows() && "A.cols() != points.rows()");
		assert(A.rows() == b.rows() && "A.rows() != b.rows()");
		m_boundary_points.reserve(static_cast<size_t>(points.rows()) + 1);
		m_rank_A_ub  = A.rows();
		m_dim_points = points.rows();
	}

	// Compute the ball of minimum radius that bounds the points in X_idx
	// and contains the points of m_boundary_points on its boundary, while respecting
	// the affine constraints present in helper.
	auto solve(vector<Index>& X_idx)
		-> tuple<RealVector<SolutionExactType>, SolutionExactType, bool> {
		if (X_idx.size() == 0 || subspace_rank_lb() == 0) {
			// If there are no points to bound or if the constraints are likely to
			// determine a unique point, then compute the point of minimum norm
			// that satisfies the constraints.
			return solve_intermediate();
		}
		// Find the constrained miniball of all except the last point.
		Index i = X_idx.back();
		X_idx.pop_back();
		auto&& [centre, sqRadius, success] = solve(X_idx);
		auto&& sqDistance =
			(m_points.col(i).template cast<SolutionExactType>() - centre).squaredNorm();
		if (sqDistance > sqRadius) {
			// If the last point does not lie in the computed bounding ball,
			// add it to the list of points that will lie on the boundary of the
			// eventual ball. This determines a new constraint.
			add_point(i);
			// compute a bounding ball with the new constraint
			std::tie(centre, sqRadius, success) = solve(X_idx);
			// Undo the addition of the last point.
			// This matters in nested calls to this function
			// because we assume that the function does not mutate its arguments.
			remove_last_point();
		}
		X_idx.push_back(i);
		return tuple{centre, sqRadius, success};
	}
};
}  // namespace detail

namespace utility {
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
// Class to perform weakly monotonic conversion from CGAL::Quotient<CGAL::Gmpzf> to double.
class ToDouble {
	mpq_t m_value;

  public:
	ToDouble() {  // NOLINT(cppcoreguidelines-pro-type-member-init)
		mpq_init(m_value);
	}

	auto operator()(const CGAL::Quotient<CGAL::Gmpzf>& x) -> double {
		const CGAL::Gmpzf& num = x.numerator();
		const CGAL::Gmpzf& den = x.denominator();

		auto num_man = num.man();
		auto num_exp = num.exp();

		auto den_man = den.man();
		auto den_exp = den.exp();

		// The value is (num.man / den.man) * 2^(num.exp - den.exp).
		// We compute this using GMP's rational numbers for exactness.

		mpq_set_num(m_value, num_man);
		mpq_set_den(m_value, den_man);

		auto exp_diff = num_exp - den_exp;

		if (exp_diff > 0) {
			mpq_mul_2exp(m_value, m_value, exp_diff);
		} else if (exp_diff < 0) {
			mpq_div_2exp(m_value, m_value, -exp_diff);
		}

		double result = mpq_get_d(m_value);

		return result;
	}

	~ToDouble() {
		mpq_clear(m_value);
	}

	ToDouble(const ToDouble&)       = delete;
	ToDouble(ToDouble&&)            = delete;
	void operator=(const ToDouble&) = delete;
	void operator=(ToDouble&&)      = delete;
};

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
template <detail::MatrixExpr T>
auto equidistant_subspace(const T& X)
	-> std::tuple<detail::RealMatrix<typename T::Scalar>, detail::RealVector<typename T::Scalar>> {
	using detail::RealMatrix;
	using detail::RealVector;
	using std::tuple;
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

}  // namespace utility

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
-   the centre of the sphere of
minimum radius bounding every point in X.
-   the squared radius of the bounding sphere.
-   a boolean flag that is true if the solution is known to be correct.
*/
template <
	SolutionPrecision  S,
	detail::MatrixExpr X_t,
	detail::MatrixExpr A_t,
	detail::VectorExpr b_t>
	requires std::same_as<typename X_t::Scalar, typename A_t::Scalar> &&
             std::same_as<typename A_t::Scalar, typename b_t::Scalar>
auto constrained_miniball(const X_t& points, const A_t& A, const b_t& b)
	-> std::tuple<detail::RealVector<SolutionType<S>>, SolutionType<S>, bool> {
	using detail::ConstrainedMiniballSolver;
	using detail::Gmpzf;
	using detail::Index;
	using detail::VectorXd;
	using std::tuple;
	using std::vector;
	using utility::ToDouble;

	using Real_t = X_t::Scalar;
	assert(A.rows() == b.rows() && "A.rows() != b.rows()");
	assert(A.cols() == points.rows() && "A.cols() != X.rows()");
	vector<Index> X_idx(points.cols());
	std::iota(X_idx.begin(), X_idx.end(), 0);
	// shuffle the points
	std::shuffle(X_idx.begin(), X_idx.end(), std::random_device());

	// Get the result
	ConstrainedMiniballSolver solver(
		points.template cast<Gmpzf>(),
		A.template cast<Gmpzf>(),
		b.template cast<Gmpzf>()
	);
	if constexpr (S == SolutionPrecision::EXACT) {
		return solver.solve(X_idx);
	} else {
		auto to_double                   = ToDouble();
		auto [centre, sqRadius, success] = solver.solve(X_idx);
		VectorXd centre_d(points.rows());
		for (int i = 0; i < points.rows(); i++) {
			centre_d[i] = to_double(centre(i));
		}
		double sqRadius_d = to_double(sqRadius);
		return tuple{centre_d, sqRadius_d, success};
	}
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
-   the centre of the sphere of
minimum radius bounding every point in X.
-   the squared radius of the bounding sphere.
-   a boolean flag that is true if the solution is known to be correct
*/
template <SolutionPrecision S, detail::MatrixExpr X_t>
auto miniball(const X_t& X)
	-> std::tuple<detail::RealVector<SolutionType<S>>, SolutionType<S>, bool> {
	using detail::Matrix;
	using detail::Vector;
	using Real_t = X_t::Scalar;
	using Mat    = Matrix<Real_t, Eigen::Dynamic, Eigen::Dynamic>;
	using Vec    = Vector<Real_t, Eigen::Dynamic>;
	return constrained_miniball<S>(X, Mat(0, X.rows()), Vec(0));
}

}  // namespace cmb

#endif  // CMB_HPP
