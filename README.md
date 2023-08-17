# Constrained Smallest Enclosing Ball Algorithm
This is a C++ implementation of Emo Welzl's algorithm [[1]](#bib1) to find the smallest bounding ball of a point set in Euclidean space, with modifications to allow linear constraints on the centre of the bounding ball. 

Given $X = \{x_1, \ldots, x_n\} \subset \mathbb{R}^d$ and an affine subspace $E \subset \mathbb{R}^d$, we wish to find
```math
\mathrm{argmin}_{z \in E} \ \max \{\| z - x_1\|_2, \ldots, \|z - x_n\|_2 \}
```
i.e., the centre of the smallest bounding ball of $X$ whose centre is constrained to lie in $E$.

The problem can be solved in amortised $O(n)$ time by using Welzl's algorithm, with a modification of the terminating condition of Welzl's algorithm, noting that the number of points defining a sphere with centre in $E$ depends on the arrangement of points and $E$, and might be less than $d+1$. 

## Installation and requirements
The algorithm is provided as a single header-only library, and requires
- C++20 compliant compiler with support for concepts (GCC 10.3 or later or later, MSVC 2019 16.3 or later).
- The [Eigen C++ library](https://eigen.tuxfamily.org/index.php?title=Main_Page) (tested with version 3.4.0).

### Optional Dependencies
- The [GNU MP Library](https://gmplib.org/) (tested with version 6.3.0) and the [GNU MPFR Library](https://www.mpfr.org/) (tested with version 4.2.0) for exact geometric computation. These will automatically be used if `<mpfr.h>` is found. You can disable this behaviour by defining the preprocessor macro `CMB_NO_MPFR`.

## Example usage
See `example.cpp` for examples.

## References

<a name="bib1">[1]</a> E. Welzl, “Smallest enclosing disks (balls and ellipsoids),” in New Results and New Trends in Computer Science, H. Maurer, Ed., in Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 1991, pp. 359–370. doi: 10.1007/BFb0038202.

