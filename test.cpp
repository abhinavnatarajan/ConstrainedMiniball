#include "ConstrainedMiniball.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <tuple>
#include <vector>
#include <cassert>

using 
    std::cout, 
    std::endl, 
    std::cin, 
    std::vector, 
    std::tie,
    std::tuple,
    std::sqrt, 
    std::min, 
    std::abs, 
    std::sin, 
    std::cos, 
    std::numbers::pi;

template <typename T>
using limits = std::numeric_limits<T>;

using 
    Eigen::Dynamic,
    Eigen::NoChange,
    Eigen::all, 
    Eigen::MatrixBase;

template <typename T>
using Vec = Eigen::Vector<T, Dynamic>;

template <typename T>
using Mat = Eigen::Matrix<T, Dynamic, Dynamic>;

using 
    cmb::constrained_miniball,
    cmb::RealMatrix, 
    cmb::RealVector;

template <class Derived>
tuple<
    RealMatrix<typename Derived::Scalar>, 
    RealVector<typename Derived::Scalar>> 
    equidistant_subspace(const MatrixBase<Derived>& X) {
    int n = X.cols();
    typedef Derived::Scalar Real_t;
    RealMatrix<Real_t> E(n-1, X.rows());
    RealVector<Real_t> b(n-1);
    if (n > 1) {
        b = 0.5 * (X.rightCols(n-1).colwise().squaredNorm().array() - X.col(0).squaredNorm()).transpose();
        E = (X.rightCols(n-1).colwise() - X.col(0)).transpose();
    }
    return tuple{E, b};
}

template <typename T>
bool approx_equal(const T& a, const T& b)
{
    static constexpr T eps = static_cast<T>(1e-4);
    static constexpr T abs_eps = static_cast<T>(1e-12);
    if (a != static_cast<T>(0) && b != static_cast<T>(0))
    {
        return (abs(a - b) <= eps * min(a, b));
    }
    else 
    {
        return (abs(a - b) <= abs_eps);
    }
    
}

int main() {
    typedef double S;
    cout << "Using epsilon : " << limits<S>::epsilon() << endl;
    cout << "Using Eigen tolerance : " << Eigen::NumTraits<S>::dummy_precision() << endl;

    int i = 0;
    cout << "Test " << ++i << endl;
    cout << "------------------------" << endl;
    // 3 equidistant points on the unit circle in the xy-plane in 3D
    Mat<S> X {
        {1.0, -0.5, -0.5},
        {0.0, sin(2 * pi / 3), sin(4 * pi / 3)},
        {0.0, 0.0, 0.0}
    }, 
    // Ax = b define the z=1 plane
    A {
        {0.0, 0.0, 1.0}
    };
    Vec<S> b { {1.0} };
    Vec<S> correct_centre{ {0.0, 0.0, 1.0 }};
    S correct_sqRadius(2.0);
    auto [centre, sqRadius, success] = constrained_miniball<S>(X, A, b);

    cout << "X : " << endl;
    cout << X << endl;
    cout << "A : " << endl;
    cout << A << endl;
    cout << "b^T : " << b.transpose().eval() << endl;
    cout << "Solution found: " << (success ? "true" : "false") << endl;
    cout << "Centre : " << centre.transpose().eval() << endl;
    cout << "Squared radius : " << sqRadius << endl;
    cout << "Expected centre : " << correct_centre.transpose().eval() << endl;
    cout << "Expected squared radius : " << correct_sqRadius << endl;
    cout << "Delta radius : " << abs(sqRadius - correct_sqRadius) << endl;
    cout << "Delta centre (squared norm) : " << (centre - correct_centre).norm() << endl;
    assert(approx_equal((centre - correct_centre).norm(), static_cast<S>(0)));
    assert(approx_equal(sqRadius, correct_sqRadius));
    assert(success);

    // Try an edge case 
    // Same points in 2D
    cout << "Test " << ++i << endl;
    cout << "------------------------" << endl;
    X.conservativeResize(2, NoChange);
    // Set A, b to manually define the subspace equidistant from points in X
    tie(A, b) = equidistant_subspace(X);
    tie(centre, sqRadius, success) = constrained_miniball<S>(X, A, b);
    correct_sqRadius = S(1.0);
    correct_centre = Vec<S>{ {0.0, 0.0} };

    cout << "X : " << endl;
    cout << X << endl;
    cout << "A : " << endl;
    cout << A << endl;
    cout << "b^T : " << b.transpose().eval() << endl;
    cout << "Solution found: " << (success ? "true" : "false") << endl;
    cout << "Centre : " << centre.transpose().eval() << endl;
    cout << "Squared radius : " << sqRadius << endl;
    cout << "Expected centre : " << correct_centre.transpose().eval() << endl;
    cout << "Expected squared radius : " << correct_sqRadius << endl;
    cout << "Delta radius : " << abs(sqRadius - correct_sqRadius) << endl;
    cout << "Delta centre (squared norm) : " << (centre - correct_centre).norm() << endl;
    assert(approx_equal((centre - correct_centre).norm(), static_cast<S>(0)));
    assert(approx_equal(sqRadius, correct_sqRadius));
    assert(success);

    cout << "Test " << ++i << endl;
    cout << "------------------------" << endl;
    X = Mat<S> {
        {1.0, 2.0, 3.0, 2.0},
        {4.0, 3.0, 1.0, -2.0},
    };
    tie(A, b) = equidistant_subspace(X(all, vector<int>{1, 2, 3}));
    tie(centre, sqRadius, success) = constrained_miniball<S>(X, A, b);
    correct_centre = Vec<S>{ {-0.5, 0.5} };
    correct_sqRadius = S(14.5);

    cout << "X : " << endl;
    cout << X << endl;
    cout << "A : " << endl;
    cout << A << endl;
    cout << "b^T : " << b.transpose().eval() << endl;
    cout << "Solution found: " << (success ? "true" : "false") << endl;
    cout << "Centre : " << centre.transpose().eval() << endl;
    cout << "Squared radius : " << sqRadius << endl;
    cout << "Expected centre : " << correct_centre.transpose().eval() << endl;
    cout << "Expected squared radius : " << correct_sqRadius << endl;
    cout << "Delta radius : " << abs(sqRadius - correct_sqRadius) << endl;
    cout << "Delta centre (squared norm) : " << (centre - correct_centre).norm() << endl;
    assert(approx_equal((centre - correct_centre).norm(), static_cast<S>(0)));
    assert(approx_equal(sqRadius, correct_sqRadius));
    assert(success);

    cout << "Test " << ++i << endl;
    cout << "------------------------" << endl;
    tie(A, b) = equidistant_subspace(X(all, vector<int>{2, 3}));
    tie(centre, sqRadius, success) = constrained_miniball<S>(X, A, b);
    correct_centre = Vec<S>{ {-0.2, 0.4} };
    correct_sqRadius = S(14.4);

    cout << "X : " << endl;
    cout << X << endl;
    cout << "A : " << endl;
    cout << A << endl;
    cout << "b^T : " << b.transpose().eval() << endl;
    cout << "Solution found: " << (success ? "true" : "false") << endl;
    cout << "Centre : " << centre.transpose().eval() << endl;
    cout << "Squared radius : " << sqRadius << endl;
    cout << "Expected centre : " << correct_centre.transpose().eval() << endl;
    cout << "Expected squared radius : " << correct_sqRadius << endl;
    cout << "Delta radius : " << abs(sqRadius - correct_sqRadius) << endl;
    cout << "Delta centre (squared norm) : " << (centre - correct_centre).norm() << endl;
    assert(approx_equal((centre - correct_centre).norm(), static_cast<S>(0)));
    assert(approx_equal(sqRadius, correct_sqRadius));
    assert(success);
    
    int t;
    cin >> t;
    return 0;
}