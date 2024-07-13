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
#include "ConstrainedMiniball.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numbers>
#include <tuple>

using std::cout, std::endl, std::cin;
using namespace cmb;

template <class Derived>
tuple<RealMatrix<typename Derived::Scalar>,
	  RealVector<typename Derived::Scalar>>
equidistant_subspace(const Eigen::MatrixBase<Derived> &X) {
	int n = X.cols();
	typedef typename Derived::Scalar Real_t;
	RealMatrix<Real_t> E(n - 1, X.rows());
	RealVector<Real_t> b(n - 1);
	if (n > 1) {
		b = 0.5 * (X.rightCols(n - 1).colwise().squaredNorm().array() -
				   X.col(0).squaredNorm())
					  .transpose();
		E = (X.rightCols(n - 1).colwise() - X.col(0)).transpose();
	}
	return tuple{E, b};
}

int main() {
	// 3 equidistant points on the unit circle in the xy-plane in 3D
	Eigen::MatrixXd X{{1.0, -0.5, -0.5},
					  {0.0, std::sin(2 * std::numbers::pi / 3),
					   std::sin(4 * std::numbers::pi / 3)},
					  {0.0, 0.0, 0.0}},
		// Ax = b define the z=1 plane
		A{{0.0, 0.0, 1.0}};
	Eigen::VectorXd b{{1.0}};
	auto [centre, sqRadius, success] =
		cmb::constrained_miniball<cmb::SolverMethod::PSEUDOINVERSE>(X, A, b);
	cout << "Solution found: " << (success ? "true" : "false") << endl;
	cout << "Centre : " << centre.transpose().eval() << endl;
	cout << "Squared radius : " << sqRadius << endl;

	// Try an edge case
	// Same points in 2D
	X.conservativeResize(2, Eigen::NoChange);
	// Set A, b to manually define the subspace equidistant from points in X
	std::tie(A, b) = equidistant_subspace(X);
	std::tie(centre, sqRadius, success) =
		cmb::constrained_miniball<cmb::SolverMethod::QP_SOLVER>(X, A, b);
	cout << "Solution found: " << (success ? "true" : "false") << endl;
	cout << "Centre : " << centre.transpose().eval() << endl;
	cout << "Squared radius : " << sqRadius << endl;

	int t;
	cin >> t;
	return 0;
}
