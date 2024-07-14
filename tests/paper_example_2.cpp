#include "test_utils.hpp"

#include <vector>

int main() {
	// Test 3
	using cmb::equidistant_subspace, cmb::test::execute_test;
	const Eigen::MatrixXd X{
		{1.0, 2.0, 3.0,  2.0},
		{4.0, 3.0, 1.0, -2.0},
	};
	const auto [A, b] = equidistant_subspace(X(Eigen::all, std::vector<int>{2, 3}));
	const Eigen::VectorXd correct_centre{
		{-0.2, 0.4}
    };
	const double correct_sqRadius{14.4};
	execute_test(X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
