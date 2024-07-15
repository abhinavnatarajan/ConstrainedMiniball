#include "test_utils.hpp"

#include <vector>

int main(int argc, char* argv[]) {
	using cmb::test::start_test;

	// Test params
	using cmb::equidistant_subspace;
	const Eigen::MatrixXd X{
		{1.0, 2.0, 3.0,  2.0},
		{4.0, 3.0, 1.0, -2.0},
	};
	const auto [A, b] = equidistant_subspace(X(Eigen::all, std::vector<int>{1, 2, 3}));
	const Eigen::VectorXd correct_centre{
		{-0.5, 0.5}
    };
	const double correct_sqRadius{14.5};
	start_test(argc, argv, X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
