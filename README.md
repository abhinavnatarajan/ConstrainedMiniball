# Constrained Smallest Enclosing Ball Algorithm
CMB (<u>C</u>onstrained <u>M</u>ini <u>B</u>all) is a C++ library to compute minimum bounding balls with affine constraints on the centre of the ball. CMB implements a modified version of Emo Welzl's Miniball algorithm [[1]](#bib1). 

Given $X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d$ and an affine subspace $E \subset \mathbb{R}^d$, we wish to find
```math
\mathrm{argmin}_{z \in E} \ \max \{\| z - x_1\|_2, \ldots, \|z - x_n\|_2 \}
```
i.e., the centre of the smallest bounding ball of $X$ whose centre is constrained to lie in $E$.

The problem can be solved in amortized $O(n)$ time by using Welzl's algorithm, with a modification of the terminating condition of Welzl's algorithm.

## Installation and requirements
CMB is provided as a single header-only library `cmb.hpp`, and requires
- C++20 compliant compiler with support for concepts (GCC 10.3, Clang 10, MSVC 2019 16.3 or later versions of these compilers).
- The [Eigen C++ library](https://eigen.tuxfamily.org/index.php?title=Main_Page) (tested with version 3.4.0).
- The [Computational Geometry Algorithms Library](https://cgal.org/) (tested with version 5.6.0).
- The [GNU Multiple Precision library](https://gmplib.org/) (tested with version 6.2.1).

## Example usage
See `example.cpp` for examples.

## Running the tests
The tests in the `test` folder can be built with CMake by running `cmake . && cmake --build` in the root folder. The resulting test program will be in the build folder under the name `runtests`.

## License
Copyright (c) 2023 Abhinav Natarajan.

ConstrainedMiniball is released under the GNU General Public License ("GPL").

## References

<a name="bib1">[1]</a> E. Welzl, “Smallest enclosing disks (balls and ellipsoids),” in New Results and New Trends in Computer Science, H. Maurer, Ed., in Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 1991, pp. 359–370. doi: 10.1007/BFb0038202.
