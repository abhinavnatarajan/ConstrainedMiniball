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
#include "ConstrainedMiniball.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <numbers>

using std::cout, std::endl, std::cin;

int main() {
    Eigen::MatrixXd X {
        {1.0, -0.5, -0.5},
        {0.0, std::sin(2 * std::numbers::pi / 3), std::sin(4 * std::numbers::pi / 3)},
        {0.0, 0.0, 0.0}
    }, 
    A {
        {0.0, 0.0, 1.0}
    };
    Eigen::VectorXd b { {1.0} };
    auto [centre, sqRadius, success] = cmb::constrained_miniball(3, X, A, b);
    cout << "Solution found: " << (success ? "true" : "false") << endl;
    cout << "Centre : " << centre.transpose().eval() << endl;
    cout << "Squared radius : " << sqRadius << endl;
    int t;
    cin >> t;
    return 0;
}