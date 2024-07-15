#include "test_utils.hpp"

#include <Eigen/Dense>

#include <CGAL/Gmpzf.h>

int main(int argc, char* argv[]) {
	using cmb::test::start_test;
	using std::cerr, std::endl;

	// Test params
	using std::sin, std::numbers::pi;
	cerr << "Points: 3 points exactly on the unit circle in the z=0 plane in 3D" << endl;
	cerr << "Constraints: z=1 plane" << endl;
	using MatrixXe = Eigen::Matrix<CGAL::Gmpzf, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorXe = Eigen::Vector<CGAL::Gmpzf, Eigen::Dynamic>;
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
	const CGAL::Gmpzf correct_sqRadius(2.0);
	start_test(argc, argv, X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
