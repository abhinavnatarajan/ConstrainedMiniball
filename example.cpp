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

    GNU General Public License ("GPL") copyright permissions statement:
    **************************************************************************
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include "cmb.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <numbers>

using Eigen::MatrixX, Eigen::MatrixXd, Eigen::VectorX, Eigen::VectorXd;
using std::cout, std::getchar, std::numeric_limits, std::setprecision;

void print_exact_vec(const VectorX<cmb::SolutionExactType>& vec) {
	for (const auto & i : vec) {
		cout << i << ' ';
	}
}

auto main() -> int {
	// 3 equidistant points on the unit circle in the xy-plane in 3D
	MatrixXd X{
		{1.0,							   -0.5,							   -0.5},
		{0.0, std::sin(2 * std::numbers::pi / 3), std::sin(4 * std::numbers::pi / 3)},
		{0.0,								0.0,								0.0}
	},
		// Ax = b define the z=1 plane
		A{{0.0, 0.0, 1.0}};
	Eigen::VectorXd b{{1.0}};
	auto [centre, sqRadius, success] =
		cmb::constrained_miniball<cmb::SolutionPrecision::DOUBLE>(X, A, b);
	cout << setprecision(numeric_limits<double>::digits10); // Set precision for output
	cout << "Solution found: " << (success ? "true" : "false") << '\n';
	cout << "Centre : " << centre.transpose().eval() << '\n';
	cout << "Squared radius : " << sqRadius << '\n';

	// Try an edge case
	// Same points in 2D
	X.conservativeResize(2, Eigen::NoChange);
	// Set A, b to manually define the subspace equidistant from points in X
	std::tie(A, b) = cmb::utility::equidistant_subspace(X);
	auto&& [centre2, sqRadius2, success2] =
		cmb::constrained_miniball<cmb::SolutionPrecision::EXACT>(X, A, b);
	cout << "Solution found: " << (success2 ? "true" : "false") << '\n';
	cout << "Centre : ";
	print_exact_vec(centre2);
	cout << '\n';
	cout << "Squared radius : " << sqRadius2 << '\n';

	getchar();
	return 0;
}
