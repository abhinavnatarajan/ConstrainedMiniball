#include "test_utils.hpp"

#include <Eigen/Dense>

#include <CGAL/Mpzf.h>

auto main(int argc, char* argv[]) -> int {
	using cmb::test::start_test;
	using std::cerr, std::endl;

	// Test params
	using std::sin, std::numbers::pi;
	cerr << "Points: 3 points exactly on the unit circle in the z=0 plane in 3D" << '\n';
	cerr << "Constraints: z=1 plane" << '\n';
	using MatrixXe = Eigen::Matrix<CGAL::Mpzf, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorXe = Eigen::Vector<CGAL::Mpzf, Eigen::Dynamic>;
	const MatrixXe X{
		{1.0, 0.0, -1.0},
		{0.0, 1.0,  0.0},
		{0.0, 0.0,  0.0}
	};
	const MatrixXe A{
		{0.0, 0.0, 1.0}
	};
	const VectorXe b{{1.0}};
	const VectorXe correct_centre{
		{0.0, 0.0, 1.0}
	};
	const CGAL::Mpzf correct_sqRadius(2.0);
	start_test(argc, std::span<char*>(argv, argc), X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
