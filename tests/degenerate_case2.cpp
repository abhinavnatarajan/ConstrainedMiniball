#include "test_utils.hpp"

#include <cmath>
#include <numbers>

auto main(int /*argc*/, char* /*argv*/[]) -> int {
	using std::cerr, std::endl;

	// Test params
	using std::sin, std::numbers::pi;
	cerr << "Point set: 3 points forming a right-angled triangle in 2D." << '\n';
	const Eigen::MatrixXd X{
		{33635.41277951287, 33633.44454288007, 33618.487612377154},
		{8450.704467005053, 8429.796907602124,  8452.297801422084},
	};

	cmb::utility::TypeConverter<cmb::SolutionExactType, double> to_double;
	auto&& [centre, sq_radius, success] = cmb::miniball<cmb::SolutionPrecision::EXACT>(X);
	cerr << std::setprecision(std::numeric_limits<double>::max_digits10) << '\n';
	// cerr << "Centre of the bounding sphere of the triangle: " << centre.template
	// cast<double>().transpose() << '\n';
	cerr << "Squared radius of the bounding sphere of the triangle: "
		 << to_double.get_mpq_t(sq_radius) << " (exact) " << to_double(sq_radius) << " (double)"
		 << '\n';

	auto&& [centre1, sq_radius1, success1] =
		cmb::miniball<cmb::SolutionPrecision::EXACT>(X(Eigen::all, {1, 2}));
	// cerr << "Centre of the bounding sphere of the diameter line: " << centre1.template
	// cast<double>().transpose() << '\n';
	cerr << "Squared radius of the bounding sphere of the diameter line: "
		 << to_double.get_mpq_t(sq_radius1) << " (exact) " << to_double(sq_radius1) << " (double)"
		 << '\n';

	assert(
		sq_radius1 <= sq_radius &&
		"Bounding sphere of the diameter line should be smaller than the triangle's bounding sphere"
	);
	return 0;
}
