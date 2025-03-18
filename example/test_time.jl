using LinearAlgebra, JacobiEigen, Plots, GenericLinearAlgebra, Quadmath

########################################################################
# Adapt randsvd in MatrixDepot.jl such that it can generate SPD matrices with pre-defined singular values. 

function qmult!(A::Matrix{T}) where T
    n, m = size(A)

    d = zeros(T, n)
    for k = n-1:-1:1

        # generate random Householder transformation
        x = randn(n-k+1)
        s = norm(x)
        sgn = sign(x[1]) + (x[1]==0)
        s = sgn * s
        d[k] = -sgn
        x[1] = x[1] + s
        beta = s * x[1]

        # apply the transformation to A
        y = x'*A[k:n, :];
        A[k:n, :] = A[k:n, :] - x * (y /beta)
    end

    # tidy up signs
    for i=1:n-1
        A[i, :] = d[i] * A[i, :]
    end
    A[n, :] = A[n, :] * sign(randn())
    return A
end

function randsvd(::Type{T}, m::Integer, n::Integer, kappa, mode::Integer) where T
    abs(kappa) >= 1 || throw(ArgumentError("Condition number must be at least 1."))
    kappa = convert(T, kappa)

    if kappa < 0
        kappa = -1 * kappa; 
        pd = true; 
    else
        pd = false; 
    end 

    p = min(m,n)
    if p == 1 # handle 1-d case
        return ones(T, 1, 1)*kappa
    end

    if mode == 3
        factor = kappa^(-1/(p-1))
        sigma = factor.^[0:p-1;]
    elseif mode == 4
        sigma = ones(T, p) - T[0:p-1;]/(p-1)*(1 - 1/kappa)
    elseif mode == 5
        sigma = exp.(-rand(p) * log(kappa))
    elseif mode == 2
        sigma = ones(T, p)
        sigma[p] = one(T)/kappa
    elseif mode == 1
        sigma = ones(p)./kappa
        sigma[1] = one(T)
    else
        throw(ArgumentError("invalid mode value."))
    end

    if pd 
        A = zeros(T, m, n)
        A[1:p, 1:p] = diagm(0 => sigma)
        Q = Matrix{T}(I,n,n);
        Q = qmult!(copy(Q)); 
        A = Q'*A*Q; 
        A = (A + A')/2; 
    else
        A = zeros(T, m, n)
        A[1:p, 1:p] = diagm(0 => sigma)
        A = qmult!(copy(A'))
        A = qmult!(copy(A'))
    end 

    return A
end

########################################################################    

# Function for computing forward, backward, and orthogonal errors
function ComputeError( Λᵣ, Λⱼ, Vⱼ, A )
    Λᵣ = sort(Λᵣ); 
    Λ = sort(copy(Λⱼ)); 

    # Errors 
    fwderr = maximum(abs.(Λᵣ - Λ)/abs.(Λᵣ)); 
    bwderr = norm( Vⱼ * diagm(Λⱼ) * Vⱼ' - A)/norm(A); 
    orterr = norm( Vⱼ'*Vⱼ - I );

    return fwderr, bwderr, orterr

end

# Fix matrix dimension and varying the condition number 
N = round.(10 .^ range(1, 14, length=20));
n = 200; 

fwderrm2 = zeros(Float64, length(N), 1); 
bwderrm2 = zeros(Float64, length(N), 1); 
orterrm2 = zeros(Float64, length(N), 1); 

fwderrm3 = zeros(Float64, length(N), 1); 
bwderrm3 = zeros(Float64, length(N), 1); 
orterrm3 = zeros(Float64, length(N), 1); 

fwderrj = zeros(Float64, length(N), 1); 
bwderrj = zeros(Float64, length(N), 1); 
orterrj = zeros(Float64, length(N), 1); 

tm2 = zeros(Float64, length(N), 1); 
tm3 = zeros(Float64, length(N), 1); 
tj = zeros(Float64, length(N), 1); 

for i ∈ eachindex(N)
    kappa = N[i];
    A = randsvd(Float64, n, n, -1*kappa, 3); 
    
    A1 = copy(A); 
    time1 = time(); 
    jacobi_eigen!(A1); 
    tj[i] = time() - time1; 

    A2 = copy(A); 
    time1 = time(); 
    mp2_jacobi_eigen!(A2, Float32); 
    tm2[i] = time() - time1; 

    A3 = copy(A); 
    time1 = time();
    mp3_jacobi_eigen!(A3, Float32, Float128); 
    tm3[i] = time() - time1; 
end

plt = plot(N, tm3, xscale=:log10, label="MP3")
plot!(N, tj, xscale=:log10, label="Jacobi")
plot!(N, tm2, xscale=:log10, label="MP2")

xlabel!("κ(A)")
ylabel!("Time")
display(plt)