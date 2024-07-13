#include "test_utils.hpp"

int main(int argc, char* argv[]) {
// 	// Test 3
	 MatrixXd X{
		{1.0, 2.0, 3.0,  2.0},
		{4.0, 3.0, 1.0, -2.0},
	};
	auto [A, b] = equidistant_subspace(X(all, vector<int>{2, 3}));
	VectorXd correct_centre{
		{-0.2, 0.4}
    };
	double correct_sqRadius {14.4};
	if (std::string_view(argv[1]) == "PSEUDOINVERSE") {
		execute_test<PSEUDOINVERSE>(X, A, b, correct_centre, correct_sqRadius);
	} else if (std::string_view(argv[1]) == "QP_SOLVER") {
		execute_test<QP_SOLVER>(X, A, b, correct_centre, correct_sqRadius);
	} else {
		std::cerr << "Invalid argument" << std::endl;
	}
	return 0;
}
