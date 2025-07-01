#include "../cmb.hpp"
#include <CGAL/Gmpzf.h>
#include <CGAL/Quotient.h>
#include <cmath>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
auto construct(std::stringstream& base, const int& e) -> CGAL::Gmpzf {
	CGAL::Gmpz m;
	base >> m;
	CGAL::Gmpzf result_base(m);
	CGAL::Gmpzf result_exp(pow(2, e));
	return result_base * result_exp;
}
}  // namespace

auto main() -> int {
	using CGAL::Gmpzf, CGAL::Quotient;

	std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

	// Numerator of a
	auto num_a_base = std::stringstream(
		"495465331884540240762104420639278860096116250934219950844827973437034498625"
	);  //*2^-379
	auto num_a_exp = -379;
	auto num_a     = construct(num_a_base, num_a_exp);
	std::cout << "num_a = ";
	CGAL::print(std::cout, num_a);
	std::cout << '\n';

	// Denominator of a
	auto denom_a_base = std::stringstream("2245694908428994193174821578352066558595985337089");
	auto denom_a_exp  = -299;
	auto denom_a      = construct(denom_a_base, denom_a_exp);
	std::cout << "denom_a = ";
	CGAL::print(std::cout, denom_a);
	std::cout << '\n';

	// Numerator of b
	auto num_b_base = std::stringstream("3447327498334006902041169");
	auto num_b_exp  = -215;
	auto num_b      = construct(num_b_base, num_b_exp);
	std::cout << "num_b = ";
	CGAL::print(std::cout, num_b);
	std::cout << '\n';

	// Denominator of b
	auto denom_b_base = std::stringstream("1");
	auto denom_b_exp  = -141;
	auto denom_b      = construct(denom_b_base, denom_b_exp);
	std::cout << "denom_b = ";
	CGAL::print(std::cout, denom_b);
	std::cout << '\n';

	auto a = Quotient<Gmpzf>(num_a, denom_a), b = Quotient<Gmpzf>(num_b, denom_b);
	auto to_double = cmb::utility::ToDouble();
	auto a_d = to_double(a), b_d = to_double(b);
	std::cout << "a = " << a << ", b = " << b << '\n';
	std::cout << "double(a) = " << a_d << ", double(b) = " << b_d << '\n';
	std::cout << "a > b: " << (a > b ? "True" : "False") << '\n';
	std::cout << "a_d >= b_d: " << (a_d >= b_d ? "True" : "False") << '\n';

	assert(a > b && a_d >= b_d);
}
