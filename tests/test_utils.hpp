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
#include "../cmb.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numbers>
#include <tuple>
#include <vector>

using std::cerr, std::endl, std::vector, std::tie, std::tuple, std::min, std::abs,
	std::sin, std::numbers::pi;

using Eigen::NoChange, Eigen::all, Eigen::MatrixXd, Eigen::VectorXd;

using cmb::constrained_miniball, cmb::MatrixXdExpr, cmb::SolverMethod;
using enum SolverMethod;

template <MatrixXdExpr T> tuple<MatrixXd, VectorXd> equidistant_subspace(const T& X) {
	int      n = X.cols();
	MatrixXd E(n - 1, X.rows());
	VectorXd b(n - 1);
	if (n > 1) {
		b = 0.5 * (X.rightCols(n - 1).colwise().squaredNorm().array() - X.col(0).squaredNorm())
		              .transpose();
		E = (X.rightCols(n - 1).colwise() - X.col(0)).transpose();
	}
	return tuple{E, b};
}

template <typename T> bool approx_equal(const T& a, const T& b) {
	static constexpr T eps     = static_cast<T>(1e-4);
	static constexpr T abs_eps = static_cast<T>(1e-12);
	if (a != static_cast<T>(0) && b != static_cast<T>(0)) {
		return (abs(a - b) <= eps * min(a, b));
	} else {
		return (abs(a - b) <= abs_eps);
	}
}

template <SolverMethod S>
void execute_test(MatrixXd& X,
                  MatrixXd& A,
                  VectorXd& b,
                  VectorXd& correct_centre,
                  double&   correct_sqRadius) {

	cerr << "X :" << endl;
	cerr << X << endl;
	cerr << "A :" << endl;
	cerr << A << endl;
	cerr << "b^T :" << endl;
	cerr << b.transpose().eval() << endl;
	auto [centre, sqRadius, success] = constrained_miniball<S>(X, A, b);
	cerr << "Solution found: " << (success ? "true" : "false") << endl;
	cerr << "Expected centre :" << endl;
	cerr << correct_centre.transpose().eval() << endl;
	cerr << "Centre :" << endl;
	cerr << centre.transpose().eval() << endl;
	cerr << "Delta centre (squared norm) :" << endl;
	cerr << (centre - correct_centre).squaredNorm() << endl;
	cerr << "Expected squared radius :" << endl;
	cerr << correct_sqRadius << endl;
	cerr << "Squared radius :" << endl;
	cerr << sqRadius << endl;
	cerr << "Delta radius :" << endl;
	cerr << abs(sqRadius - correct_sqRadius) << endl;
	assert(success && "Solution not found");
	assert((approx_equal((centre - correct_centre).norm(), 0.0) && "Centre not correct"));
	assert((approx_equal(sqRadius, correct_sqRadius) && "Squared radius not correct"));
	cerr << endl;
}
