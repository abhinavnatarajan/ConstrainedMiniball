#include "test_utils.hpp"

#include <cmath>
#include <numbers>

int main(int argc, char* argv[]) {
	using cmb::test::start_test;
	using std::cerr, std::endl;

	// Test params
	using std::sin, std::numbers::pi, cmb::test::run_test;
	cerr << "Point set: 3 equidistant points on the unit circle in the xy-plane in 3D" << endl;
	cerr << "Constraints: z=1 plane" << endl;
	const Eigen::MatrixXd X{
		{1.0,            -0.5,            -0.5},
		{0.0, sin(2 * pi / 3), sin(4 * pi / 3)},
		{0.0,             0.0,             0.0}
    };
	// Ax = b define the z=1 plane
	const Eigen::MatrixXd A{
		{0.0, 0.0, 1.0}
    };
	const Eigen::VectorXd b{{1.0}};
	const Eigen::VectorXd correct_centre{
		{0.0, 0.0, 1.0}
    };
	const double correct_sqRadius(2.0);
	start_test(argc, argv, X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
