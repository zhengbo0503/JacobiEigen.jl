using JacobiEigen, LinearAlgebra

n = 200;

A = randn(n,n);
A = A' * A;

A1 = copy(A);
Λᵣ, Vᵣ = eigen!(A1);

A2 = copy(A);
Λⱼ, Vⱼ, Para = jacobi_eigen!(A2);

A3 = copy(A);
Λₕ, Vₕ, Paras = mp_jacobi_eigen(A3); 

println("Forward Error = $(maximum(abs.(sort(Λₕ) - sort(Λᵣ))./abs.(sort(Λᵣ))))");
println("Backward Error = $(norm(Vₕ*diagm(Λₕ)*Vₕ'-A)/norm(A))");
println("Orthogonality Error = $(norm(Vₕ'*Vₕ - I))");

# so is the new version:
@code_warntype jacobi_eigen!(A2);

# timing and memory allocation comparison:

A1 = copy(A);
@time eigen!(A1);

A2 = copy(A);
@time jacobi_eigen!(A3);

A3 = copy(A);
@time mp_jacobi_eigen(A3);


# check the memory allocation of the internal function:
tol = sqrt(n) * eps()/2
V = Matrix{Float64}(I,n,n);
J = zeros(2,2)
tmp1 = zeros(2,n)
tmp2 = zeros(2,n)
A3 = copy(A)
# only allocates 32 bytes of memory:
@time JacobiEigen._jacobi_eigen!(A3, tol, V, J, tmp1, tmp2);

# check the internal function is type stable:
@code_warntype JacobiEigen._jacobi_eigen!(A3, tol, V, J, tmp1, tmp2);