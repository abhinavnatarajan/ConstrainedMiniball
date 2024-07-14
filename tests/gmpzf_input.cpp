#include "test_utils.hpp"

#include <Eigen/Dense>

#include <CGAL/Gmpzf.h>

int main() {
	using cmb::test::execute_test, std::sin, std::numbers::pi;
	using MatrixXe = Eigen::Matrix<CGAL::Gmpzf, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorXe = Eigen::Vector<CGAL::Gmpzf, Eigen::Dynamic>;
	const MatrixXe X{
		{1.0,            -0.5,            -0.5},
		{0.0, sin(2 * pi / 3), sin(4 * pi / 3)},
		{0.0,             0.0,             0.0}
    };
	const MatrixXe A{
	{ 0.0, 0.0, 1.0 }
	};
	const VectorXe b{{1.0}};
	const VectorXe correct_centre{
		{0.0, 0.0, 1.0}
    };
	const CGAL::Gmpzf correct_sqRadius(2.0);
	execute_test(X, A, b, correct_centre, correct_sqRadius);
	return 0;
}
