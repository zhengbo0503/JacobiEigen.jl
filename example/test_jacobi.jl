include("Jacobi.jl")
include("JacobiEigenvalueAlgorithm.jl")

# the dot is used below to tell Julia to use the version of JacobiEigenvalueAlgorithm.jl that is 
# in the current directory, rather than in the package manager
using .JacobiEigenvalueAlgorithm

n = 500;

A = randn(n,n);
A = A' * A;

A1 = copy(A);
Λⱼ, Vⱼ, Para = Jacobi(A1)

A2 = copy(A);
Λᵣ, Vᵣ = eigen!(A2);

A3 = copy(A);
Λⱼ, Vⱼ, Para = jacobi_eigen!(A3)

println("Forward Error = $(maximum(abs.(sort(Λⱼ) - sort(Λᵣ))./abs.(sort(Λᵣ))))");
println("Backward Error = $(norm(Vⱼ*diagm(Λⱼ)*Vⱼ'-A)/norm(A))");
println("Orthogonality Error = $(norm(Vⱼ'*Vⱼ - I))");

# the original code is type stable (i.e. the compiler can work out what type all the variables are):
@code_warntype Jacobi(A1);

# so is the new version:
@code_warntype jacobi_eigen!(A3);

# timing and memory allocation comparison:
# many Gigabytes of memory is created
A1 = copy(A);
@time Jacobi(A1);

A2 = copy(A);
@time eigen!(A2);

A3 = copy(A);
@time jacobi_eigen!(A3);


# check the memory allocation of the internal function:
tol = sqrt(n) * eps()/2
V = Matrix{Float64}(I,n,n)
J = zeros(2,2)
tmp1 = zeros(2,n)
tmp2 = zeros(2,n)
A3 = copy(A)
# only allocates 32 bytes of memory:
@time JacobiEigenvalueAlgorithm._jacobi_eigen!(A3, tol, V, J, tmp1, tmp2);

# check the internal function is type stable:
@code_warntype JacobiEigenvalueAlgorithm._jacobi_eigen!(A3, tol, V, J, tmp1, tmp2);