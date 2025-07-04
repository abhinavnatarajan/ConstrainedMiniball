#include "../cmb.hpp"
#include <CGAL/Gmpq.h>
#include <CGAL/Mpzf.h>
#include <CGAL/Quotient.h>
#include <cmath>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
auto construct(std::stringstream& base_str, const long& e) -> CGAL::Mpzf {
	mpz_t m;
	mpz_init(m);
	base_str >> m;
	if (e > 0) {
		mpz_mul_2exp(m, m, e);
		CGAL::Mpzf result;
		result.init_from_mpz_t(m);
		mpz_clear(m);
		return result;
	} else {
		mpz_t exp;
		mpz_init(exp);
		mpz_set_ui(exp, 1);
		mpz_mul_2exp(exp, exp, -e);
		CGAL::Mpzf mantissa(m);
		CGAL::Mpzf exponent_inv(exp);
		mpz_clear(m);
		mpz_clear(exp);
		return mantissa / exponent_inv;
	}
}

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
}  // namespace

auto main() -> int {
	using CGAL::Mpzf, CGAL::Quotient;
	using cmb::utility::TypeConverter;

	std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

	// Numerator of a
	TypeConverter<Mpzf, mpq_class>                Mpzf_to_mpq;
	TypeConverter<cmb::SolutionExactType, double> Quotient_to_double;
	auto num_a_base_str = std::stringstream(
		"495465331884540240762104420639278860096116250934219950844827973437034498625"
	);  //*2^-379
	auto num_a_exp = -379;
	auto num_a     = construct(num_a_base_str, num_a_exp);
	std::cout << "num_a = " << Mpzf_to_mpq(num_a) << '\n';

	// Denominator of a
	auto denom_a_base = std::stringstream("2245694908428994193174821578352066558595985337089");
	auto denom_a_exp  = -299;
	auto denom_a      = construct(denom_a_base, denom_a_exp);
	std::cout << "denom_a = " << Mpzf_to_mpq(denom_a) << '\n';

	// Numerator of b
	auto num_b_base = std::stringstream("3447327498334006902041169");
	auto num_b_exp  = -215;
	auto num_b      = construct(num_b_base, num_b_exp);
	std::cout << "num_b = " << Mpzf_to_mpq(num_b) << '\n';

	// Denominator of b
	auto denom_b_base = std::stringstream("1");
	auto denom_b_exp  = -141;
	auto denom_b      = construct(denom_b_base, denom_b_exp);
	std::cout << "denom_b = " << Mpzf_to_mpq(denom_b) << '\n';

	auto a = cmb::SolutionExactType(num_a, denom_a), b = cmb::SolutionExactType(num_b, denom_b);
	auto a_d = Quotient_to_double(a), b_d = Quotient_to_double(b);
	std::cout << "a = " << Quotient_to_double.get_mpq_t(a) << ", b = " << Quotient_to_double.get_mpq_t(b) << '\n';
	std::cout << "double(a) = " << a_d << ", double(b) = " << b_d << '\n';
	std::cout << "a > b: " << (a > b ? "True" : "False") << '\n';
	std::cout << "a_d >= b_d: " << (a_d >= b_d ? "True" : "False") << '\n';
	assert(a > b && a_d >= b_d);
}
