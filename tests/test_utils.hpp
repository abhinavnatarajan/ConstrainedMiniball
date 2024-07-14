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
#include <tuple>
#include <vector>

using std::cerr, std::endl;

namespace cmb {
namespace test {
using std::tie, std::tuple, std::vector, std::min, std::abs;

using Eigen::NoChange, Eigen::all, Eigen::MatrixXd, Eigen::VectorXd;

template <typename T>
bool approx_equal(const T& a, const T& b, const T& rel_tol, const T& abs_tol) {
	using namespace detail;
	if (a != static_cast<T>(0) && b != static_cast<T>(0)) {
		return (abs(a - b) <= rel_tol * min(a, b));
	} else {
		return (abs(a - b) <= abs_tol);
	}
}

template <class T>
void execute_test(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X,
                  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
                  const Eigen::Vector<T, Eigen::Dynamic>&                 b,
                  const Eigen::Vector<T, Eigen::Dynamic>&                 correct_centre,
                  const T&                                                correct_sqRadius) {
	// cerr << "X :" << endl;
	// cerr << X << endl;
	// cerr << "A :" << endl;
	// cerr << A << endl;
	// cerr << "b^T :" << endl;
	// cerr << b.transpose().eval() << endl;
	auto [centre, sqRadius, success] = constrained_miniball(X, A, b);
	cerr << "Solution found : " << (success ? "true" : "false") << endl;
	// cerr << "Expected centre :" << endl;
	// cerr << correct_centre.transpose().eval() << endl;
	// cerr << "Centre :" << endl;
	// cerr << centre.transpose().eval() << endl;
	cerr << "Error in centre (squared norm) :" << endl;
	SolutionScalarType err_centre =
		(centre - correct_centre.template cast<SolutionScalarType>()).squaredNorm();
	cerr << err_centre << endl;
	cerr << "Expected squared radius :" << endl;
	cerr << correct_sqRadius << endl;
	cerr << "Squared radius :" << endl;
	cerr << sqRadius << endl;
	cerr << "Squared radius error :" << endl;
	SolutionScalarType err_radius =
		abs(sqRadius - static_cast<SolutionScalarType>(correct_sqRadius));
	cerr << err_radius << endl;
	assert(success && "Solution not found");
	const SolutionScalarType& rel_tol = static_cast<SolutionScalarType>(1e-4);
	const SolutionScalarType& abs_tol = static_cast<SolutionScalarType>(1e-12);
	const SolutionScalarType  zero    = static_cast<SolutionScalarType>(0.0);
	assert((approx_equal<SolutionScalarType>(err_centre, zero, rel_tol, abs_tol) &&
	        "Centre not correct"));
	assert((approx_equal<SolutionScalarType>(err_radius, zero, rel_tol, abs_tol) &&
	        "Squared radius not correct"));
	cerr << endl;
}
}  // namespace test
}  // namespace cmb
