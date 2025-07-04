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

#include <CGAL/Quotient.h>
#include <CGAL/Mpzf.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <type_traits>

namespace cmb::test {

using Eigen::NoChange, Eigen::all, Eigen::MatrixXd, Eigen::VectorXd;

template <typename T>
auto approx_equal(const T& a, const T& b, const T& rel_tol, const T& abs_tol) -> bool {
	using std::abs, std::min;
	if (a != static_cast<T>(0) && b != static_cast<T>(0)) {
		return (abs(a - b) <= rel_tol * min(a, b));
	} else {
		return (abs(a - b) <= abs_tol);
	}
}

template <SolutionPrecision S, class T>
void run_test(
	const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X,
	const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
	const Eigen::Vector<T, Eigen::Dynamic>&                 b,
	const Eigen::Vector<T, Eigen::Dynamic>&                 correct_centre,
	const T&                                                correct_sqRadius
) {
	using soln_t  = SolutionType<S>;
	using input_t = T;
	// solution will never be downcast, only upcast
	// common_t = (soln_t <= input_t) ? input_t : soln_t;
	// where input_t is one of double, CGAL::Mpzf
	// soln_t is one of double, CGAL::Quotient<CGAL::Mpzf>
	using common_t = std::conditional_t<S == SolutionPrecision::DOUBLE, input_t, soln_t>;
	cmb::utility::TypeConverter<soln_t, common_t> soln_to_common;
	cmb::utility::TypeConverter<input_t, common_t> input_to_common;
	using std::cerr, std::endl;

	cerr << std::setprecision(std::numeric_limits<double>::max_digits10);
	auto [centre, sqRadius, success] = constrained_miniball<S>(X, A, b);

	cerr << "Solution found : " << (success ? "true" : "false") << endl;

	cerr << "Error in centre (squared norm) :" << endl;
	common_t err_centre =
		(centre.template cast<common_t>() - correct_centre.template cast<common_t>()).squaredNorm();
	cerr << err_centre << endl;

	cerr << "Expected squared radius :" << endl;
	cerr << static_cast<common_t>(correct_sqRadius) << endl;

	cerr << "Squared radius :" << endl;
	cerr << sqRadius << endl;

	cerr << "Squared radius error :" << endl;
	common_t err_radius =
		abs(static_cast<common_t>(sqRadius) - static_cast<common_t>(correct_sqRadius));
	cerr << err_radius << endl;

	assert(success && "Solution not found");
	const auto rel_tol = static_cast<common_t>(1e-4);
	const auto abs_tol = static_cast<common_t>(1e-12);
	const auto zero    = static_cast<common_t>(0.0);
	assert((approx_equal<common_t>(err_centre, zero, rel_tol, abs_tol) && "Centre not correct"));
	assert(
		(approx_equal<common_t>(err_radius, zero, rel_tol, abs_tol) && "Squared radius not correct")
	);
	cerr << endl;
}

template <class T>
void start_test(
	int                                                     argc,
	std::span<char*>                                        argv,
	const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& X,
	const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
	const Eigen::Vector<T, Eigen::Dynamic>&                 b,
	const Eigen::Vector<T, Eigen::Dynamic>&                 correct_centre,
	const T&                                                correct_sqRadius
) {
	using std::cerr, std::endl;
	if (argc > 1) {
		if (std::string_view(argv[1]) == "EXACT") {
			cerr << "Running test with exact arithmetic" << endl;
			run_test<SolutionPrecision::EXACT>(X, A, b, correct_centre, correct_sqRadius);
		} else if (std::string_view(argv[1]) == "DOUBLE") {
			cerr << "Running test with double precision arithmetic" << endl;
			run_test<SolutionPrecision::DOUBLE>(X, A, b, correct_centre, correct_sqRadius);
		} else {
			cerr << "Invalid precision specified. Use EXACT or DOUBLE\n";
		}
	} else {
		cerr << "No precision specified. Use EXACT or DOUBLE" << endl;
	}
}

}  // namespace cmb::test
