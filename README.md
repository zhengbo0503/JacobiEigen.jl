# JacobiEigen.jl

JacobiEigen.jl is a Julia package for three algorithms related to the Jacobi eigenvalue algorithm [1]:
- The classical Jacobi eigenvalue algorithm, `jacobi_eigen`.
- A 2-precision preconditioned Jacobi eigenvalue algorithm, `mp2_jacobi_eigen`. For symmetric positive definition matrices, this algorithm may be faster than the classical Jacobi algorithm, but just as accurate.
- A 3-precision preconditioned Jacobi eigenvalue algorithm, `mp3_jacobi_eigen`. For symmetric positive definition matrices, this algorithm achieves extremely high relative accuracy on each eigenvalue i.e. even for moderately ill-conditioned matrices, $|\lambda_i - \hat{\lambda}_i|/|\lambda_i|$ is a modest multiple of the working accuracy for all i.

## Usage

```julia
using GenericLinearAlgebra, Quadmath

A = randn(Float64, 500, 500)
A = A'A
Λ1, V1 = jacobi_eigen(A) # classical Jacobi
Λ2, V2 = mp2_jacobi_eigen(A, Float32) # faster
Λ3, V3 = mp3_jacobi_eigen(A, Float32, Float128) # more accurate

A = randn(Float32, 500, 500)
A = A'A
Λ1, V1 = jacobi_eigen(A) # classical Jacobi
Λ2, V2 = mp2_jacobi_eigen(A, Float16) # faster
Λ3, V3 = mp3_jacobi_eigen(A, Float16, Float64) # more accurate
```

## References

[1]  N.J. Higham, F. Tissuer, M. Webb, Z. Zhou, [Computing accurate eigenvalues using a mixed-precision Jacobi algorithm](https://arxiv.org/abs/2501.03742), submitted 2025.