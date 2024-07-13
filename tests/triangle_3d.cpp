#include "test_utils.hpp"

int main(int argc, char* argv[]) {
	// Test 1
	cerr << "Point set: 3 equidistant points on the unit circle in the xy-plane in 3D" << endl;
	cerr << "Constraints: z=1 plane" << endl;
	MatrixXd X{
		{1.0,            -0.5,            -0.5},
		{0.0, sin(2 * pi / 3), sin(4 * pi / 3)},
		{0.0,             0.0,             0.0}
    };
	// Ax = b define the z=1 plane
	MatrixXd A{
		{0.0, 0.0, 1.0}
    };
	VectorXd b{{1.0}};
	VectorXd correct_centre{
		{0.0, 0.0, 1.0}
    };
	double correct_sqRadius(2.0);
	if (std::string_view(argv[1]) == "PSEUDOINVERSE") {
		execute_test<PSEUDOINVERSE>(X, A, b, correct_centre, correct_sqRadius);
	} else if (std::string_view(argv[1]) == "QP_SOLVER") {
		execute_test<QP_SOLVER>(X, A, b, correct_centre, correct_sqRadius);
	} else {
		std::cerr << "Invalid argument" << std::endl;
	}
	return 0;
}
