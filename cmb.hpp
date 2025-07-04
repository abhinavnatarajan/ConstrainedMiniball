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

	#include <algorithm>
	#include <cstdio>
	#include <random>
	#include <tuple>
	#include <vector>

	#include <CGAL/Mpzf.h>
	#include <CGAL/QP_functions.h>
	#include <CGAL/QP_models.h>

	#include <Eigen/Dense>

namespace cmb {
using SolutionExactType = CGAL::Quotient<CGAL::Mpzf>;  // exact rational numbers

enum class SolutionPrecision : std::uint8_t {
	EXACT,  // exact rational numbers
	DOUBLE  // C++ doubles
};

template <SolutionPrecision S>
using SolutionType = std::conditional_t<S == SolutionPrecision::EXACT, SolutionExactType, double>;

namespace detail {

using CGAL::Mpzf;  // exact floats
using std::tuple, std::max, std::vector, Eigen::MatrixBase, Eigen::Matrix, Eigen::Vector,
	Eigen::MatrixXd, Eigen::VectorXd, Eigen::Index, std::same_as;

template <class Real_t, const int Rows = Eigen::Dynamic, const int Cols = Eigen::Dynamic>
using RealMatrix = Matrix<Real_t, Rows, Cols>;

template <class Real_t, const int Rows = Eigen::Dynamic>
using RealVector = RealMatrix<Real_t, Rows, 1>;

template <class Derived>
concept MatrixExpr = requires { typename MatrixBase<Derived>; };

template <class Derived>
concept VectorExpr = requires { typename MatrixBase<Derived>; } && Derived::ColsAtCompileTime == 1;

template <class Derived, class Real_t>
concept RealMatrixExpr = MatrixExpr<Derived> && same_as<typename Derived::Scalar, Real_t>;

template <class Derived, class Real_t>
concept RealVectorExpr = VectorExpr<Derived> && same_as<typename Derived::Scalar, Real_t>;

template <class T, class S>
concept CanMultiply =
	MatrixExpr<T> && MatrixExpr<S> &&
	(T::ColsAtCompileTime == S::RowsAtCompileTime || T::ColsAtCompileTime == Eigen::Dynamic ||
     S::RowsAtCompileTime == Eigen::Dynamic) &&
	same_as<typename T::Scalar, typename S::Scalar>;

using QuadraticProgram         = CGAL::Quadratic_program<Mpzf>;
using QuadraticProgramSolution = CGAL::Quadratic_program_solution<Mpzf>;

class ConstrainedMiniballSolver {
	RealMatrix<Mpzf>        m_A, m_points;  // not changed after construction
	Index                   m_rank_A_ub, m_dim_points;
	RealVector<Mpzf>        m_b;
	RealMatrix<Mpzf>        m_lhs;
	RealVector<Mpzf>        m_rhs;
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
					Mpzf(0.5) * m_lhs.bottomRows(num_point_constraints).rowwise().squaredNorm();
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
			QuadraticProgram qp(CGAL::EQUAL, false, Mpzf(0), false, Mpzf(0));
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
			QuadraticProgramSolution soln = CGAL::solve_quadratic_program(qp, Mpzf());
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
	template <RealMatrixExpr<Mpzf> points_t, RealMatrixExpr<Mpzf> A_t, RealVectorExpr<Mpzf> b_t>
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

template <typename From, typename To> class TypeConverter {};

template <typename T> class TypeConverter<T, T> {
  public:
	auto operator()(const T& x) -> T {
		return x;
	}
};

template <> class TypeConverter<SolutionExactType, double> {
	mpq_t m_numerator;
	mpq_t m_denominator;
	mpq_t m_value;

  public:
	TypeConverter() {  // NOLINT(cppcoreguidelines-pro-type-member-init)
		mpq_inits(m_numerator, m_denominator, m_value, NULL);
	}

	auto operator()(const SolutionExactType& x) -> double {
		return mpq_get_d(get_mpq_t(x));
	}

	auto get_mpq_t(const SolutionExactType& x) -> mpq_ptr {
		mpq_set_ui(m_numerator, 0, 1);
		mpq_set_ui(m_denominator, 0, 1);
		x.numerator().export_to_mpq_t(m_numerator);
		x.denominator().export_to_mpq_t(m_denominator);
		mpq_div(m_value, m_numerator, m_denominator);
		return m_value;
	}

	~TypeConverter() {
		mpq_clears(m_numerator, m_denominator, m_value, NULL);
	}

	TypeConverter(const TypeConverter&)  = delete;
	TypeConverter(TypeConverter&&)       = delete;
	void operator=(const TypeConverter&) = delete;
	void operator=(TypeConverter&&)      = delete;
};

template <> class TypeConverter<double, SolutionExactType> {
  public:
	auto operator()(const double& x) -> SolutionExactType {
		return {CGAL::Mpzf(x)};
	}
};

template <> class TypeConverter<CGAL::Mpzf, SolutionExactType> {
  public:
	auto operator()(const CGAL::Mpzf& x) -> SolutionExactType {
		return {x};
	}
};

// Useful for printing CGAL::Mpzf values exactly.
template <> class TypeConverter<CGAL::Mpzf, mpq_class> {
  public:
	auto operator()(const CGAL::Mpzf& x) -> mpq_class {
		return mpq_class(x);
	}
};

template <> class TypeConverter<CGAL::Mpzf, double> {
  public:
	auto operator()(const CGAL::Mpzf& x) -> double {
		return x.to_double();
	}
};

template <> class TypeConverter<double, CGAL::Mpzf> {
  public:
	auto operator()(const double& x) -> CGAL::Mpzf {
		return {x};
	}
};

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

template <detail::MatrixExpr T>
auto equidistant_subspace(const T& X) -> std::tuple<
	detail::RealMatrix<
		typename T::Scalar,
		(T::ColsAtCompileTime > 0 ? T::ColsAtCompileTime - 1 : Eigen::Dynamic),
		T::RowsAtCompileTime>,
	detail::RealVector<typename T::Scalar>> {
	using detail::RealMatrix;
	using detail::RealVector;
	using std::tuple;
	using Real_t        = T::Scalar;
	constexpr auto Rows = (T::ColsAtCompileTime > 0 ? T::ColsAtCompileTime - 1 : Eigen::Dynamic);
	constexpr auto Cols = T::RowsAtCompileTime;
	int            n    = X.cols();
	RealMatrix<Real_t, Rows, Cols> E(n - 1, X.rows());
	RealVector<Real_t>             b(n - 1);
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
             std::same_as<typename A_t::Scalar, typename b_t::Scalar> &&
             detail::CanMultiply<A_t, X_t> && detail::CanMultiply<A_t, b_t>
auto constrained_miniball(const X_t& points, const A_t& A, const b_t& b) -> std::
	tuple<detail::RealVector<SolutionType<S>, X_t::RowsAtCompileTime>, SolutionType<S>, bool> {
	using detail::ConstrainedMiniballSolver;
	using detail::Index;
	using detail::Mpzf;
	using detail::VectorXd;
	using std::tuple;
	using std::vector;
	using utility::TypeConverter;

	using Real_t = X_t::Scalar;
	assert(A.rows() == b.rows() && "A.rows() != b.rows()");
	assert(A.cols() == points.rows() && "A.cols() != X.rows()");
	vector<Index> X_idx(points.cols());
	std::iota(X_idx.begin(), X_idx.end(), 0);
	// shuffle the points
	std::shuffle(X_idx.begin(), X_idx.end(), std::random_device());

	// Get the result
	ConstrainedMiniballSolver solver(
		points.template cast<Mpzf>(),
		A.template cast<Mpzf>(),
		b.template cast<Mpzf>()
	);
	if constexpr (S == SolutionPrecision::EXACT) {
		return solver.solve(X_idx);
	} else {
		thread_local TypeConverter<SolutionExactType, double> to_double;
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
	using Mat    = Matrix<Real_t, 0, X_t::RowsAtCompileTime>;
	using Vec    = Vector<Real_t, X_t::RowsAtCompileTime>;
	return constrained_miniball<S>(X, Mat(0, X.rows()), Vec(0));
}

}  // namespace cmb

#endif  // CMB_HPP
