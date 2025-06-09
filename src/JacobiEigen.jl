module JacobiEigen

using LinearAlgebra 

"""
    jacobi_eigen(A::AbstractMatrix{<:AbstractFloat})
    
    Compute the spectral decomposition of a symmetric matrix A ∈ ℝⁿˣⁿ using the 
    cyclic Jacobi algorithm.

# Input arguments:
- A::Matrix : Symmetric matrix A ∈ ℝⁿˣⁿ.

# Output arguments:
- Λ : Vector Λ ∈ ℝⁿ = (λ₁,…,λₙ)
- Q : Orthogonal matrix Q ∈ ℝⁿˣⁿ such that norm(QᵀQ - I) = 𝒪(u), where u is the working
    precision.
- Params : Vector Params ∈ ℕ², where Params[1] is the number of rotation used and 
    Params[2] is the number of sweeps used. 
"""
jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}) = jacobi_eigen!(copy(A))


"""
    jacobi_eigen!(A::AbstractMatrix{<:AbstractFloat})
    
    Same as jacobi_eigen, but saves space by overwriting the input A, instead of creating a copy.
"""
function jacobi_eigen!(A::AbstractMatrix{T}) where T <: AbstractFloat
    
    # set up eigenvector matrix
    V = Matrix{T}(undef, size(A))

    # use internal function for jacobi eigenvalue algorithm
    Λ, V, Params = _jacobi_eigen!(A, V)

    # sort the eigenvalues and eigenvectors
    LinearAlgebra.sorteig!(Λ, V)
    
    return Λ, V, Params
end


function _jacobi_eigen!(A::AbstractMatrix{T}, V::AbstractMatrix{T}) where T <: AbstractFloat
    # Initialize parameters 
    n = size( A,1 )
    tol = sqrt(T(n)) * Base.eps(T)/2
    fill!(V, zero(T))
    for k ∈ axes(V,1)
        V[k,k] = one(T)
    end
    J = Matrix{T}(undef,2,2)
    tmp1 = Matrix{T}(undef,2,n)
    tmp2 = Matrix{T}(undef,2,n)
    
    # use internal function for jacobi eigenvalue algorithm
    Λ, V, Params = _jacobi_eigen!(A, tol, V, J, tmp1, tmp2)

    return Λ, V, Params
end

# Internal, in-place Jacobi algorithm. No post-processing.
function _jacobi_eigen!(A::AbstractMatrix{T}, tol::T, V::Matrix{T}, J::Matrix{T}, tmp1::Matrix{T}, tmp2::Matrix{T}) where T <: AbstractFloat
    no_sweep=0
    no_rotation=0
    done_rot = true
    while done_rot
        done_rot = false
        no_sweep = no_sweep + 1
        @inbounds for p ∈ 1:size(A,1)-1
            @inbounds for q ∈ p+1:size(A,1)
                if ( abs(A[p,q])/sqrt(abs(A[p,p]*A[q,q])) > tol  )
                    # Update parameters 
                    no_rotation = no_rotation + 1
                    done_rot = true

                    # Form the Jacobi rotation matrix 
                    ζ = ( A[q,q] - A[p,p] ) / ( 2*A[p,q] )
                    t = sign(ζ)/(abs(ζ)+sqrt(1+ζ^2))
                    cs = 1/sqrt(1+t^2)
                    sn = cs*t
                    J[1,1] = cs; J[1,2] = -sn
                    J[2,1] = sn; J[2,2] = cs

                    # Apply the Jacobi rotation
                    # A[[p,q],:] = J*A[[p,q],:]
                    @views tmp1[1,:] = A[p,:]
                    @views tmp1[2,:] = A[q,:]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views A[p,:] = tmp2[1,:]
                    @views A[q,:] = tmp2[2,:]

                    # A[:,[p,q]] = A[:,[p,q]]*J'
                    @views tmp1[1,:] = A[:,p]
                    @views tmp1[2,:] = A[:,q]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views A[:,p] = tmp2[1,:]
                    @views A[:,q] = tmp2[2,:]
                    
                    A[p,q] = 0
                    A[q,p] = 0
                    
                    # V[:,[p,q]] = J * V[:,[p,q]]
                    @views tmp1[1,:] = V[:,p]
                    @views tmp1[2,:] = V[:,q]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views V[:,p] = tmp2[1,:]
                    @views V[:,q] = tmp2[2,:]
                end
            end
        end
    end
    return diag(A), V, (no_rotation, no_sweep)
end

"""
    off(A::AbstractMatrix{<:AbstractFloat})
    
    Compute the off-quantity a matrix A ∈ ℝⁿˣⁿ:
         off(A) = sqrt(∑_{i≠j} |A[i,j]|²)
"""
function off(A::AbstractMatrix{T}) where T <: Real
    ret = norm(A)^2
    for i ∈ axes(A,1)
        ret -= abs(A[i,i])^2
    end
    return sqrt(ret)
end


"""
    mp2_jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat})

    Compute the spectral decomposition of a symmetric matrix A ∈ ℝⁿˣⁿ using the cyclic Jacobi algorithm with mixed-precision pre-processing.
    
    # Input arguments:
    - A::Matrix : Symmetric matrix A ∈ ℝⁿˣⁿ.
    - Tl::Type : Low precision type. Used to compute the eigenvectors of A to form a preconditioner.

    # Output arguments: Λ, V, Params, [timePreconditioner timeApply timeJacobi timeElse]
    - Λ::Vector : Λ ∈ ℝⁿ = (λ₁,…,λₙ) of eigenvalues sorted in ascending order.
    - V::Matrix : V ∈ ℝⁿˣⁿ = (v₁,…,vₙ) of eigenvectors.
    - Params::Tuple : (no_rotation, no_sweep) = (number of rotations, number of sweeps).
    - [timePreconditioner timeApply timeJacobi timeElse] : Timing information for each stage.

"""
mp2_jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}) = mp2_jacobi_eigen!(copy(A), Tl)

"""
    mp2_jacobi_eigen!(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat})
    
    Same as mp2_jacobi_eigen, but saves space by overwriting the input A, instead of creating a copy.
"""
function mp2_jacobi_eigen!(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}) where T <: AbstractFloat 

    # Timing for constructing the preconditioner 
    tmp = time()

    # Scale the matrix to avoid overflow 
    isscale1 = 0; 
    anrm1 = maximum(abs, A);
    eps1 = T.(Base.eps(Tl)/2); 
    safmin1 = T.(Base.floatmin(Tl));
    smlnum1 = safmin1 / eps1; 
    bignum1 = 1 / smlnum1; 
    rmin1 = sqrt( smlnum1 );
    rmax1 = sqrt( bignum1 );
    if ( anrm1 > 0 ) && ( anrm1 < rmin1 ) 
        isscale1 = 1; 
        sigma1 = rmin1 / anrm1;
    elseif ( anrm1 > rmax1 )
        isscale1 = 1; 
        sigma1 = rmax1 / anrm1;
    end
    if isscale1 == 1
        Aprime = sigma1 * copy(A); 
    else
        Aprime = copy(A); 
    end

    # Compute the low-precision eigenvectors 
    Vl = eigen!(Symmetric(Tl.(Aprime))).vectors

    # Orthogonalize the eigenvectors
    Vu = T.(Vl)
    Qu = Matrix(qr!(Vu).Q)

    # Store the time for computing the preconditioner
    timePreconditioner = time()-tmp

    # Timing for applying the preconditioner
    tmp = time()

    # Apply the preconditioner
    mul!(Vu, A, Qu)
    mul!(A, Qu', Vu)
    
    # Post-process the preconditioned matrix to make it symmetric
    hermitianpart!(A)

    # Store the time for applying the preconditioner
    timeApply = time()-tmp 

    # Timing for applying the Jacobi algorithm
    tmp = time()

    # Compute the eigensystem of the preconditioned matrix 
    Λ, Vu, Params = _jacobi_eigen!(A, Vu)

    # Store the time for applying the Jacobi algorithm
    timeJacobi = time() - tmp

    # Timing for everything else 
    tmp = time()

    # Compute the final eigenvector matrix and sort by eigenvalues
    V = A
    mul!(V, Qu, Vu)
    LinearAlgebra.sorteig!(Λ, V)

    # Store the time for everything else
    timeElse = time() - tmp

    return Λ, V, Params, [timePreconditioner, timeApply, timeJacobi, timeElse]
end


"""
    mp3_jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat})

    Compute the spectral decomposition of a symmetric matrix A ∈ ℝⁿˣⁿ using the cyclic Jacobi algorithm with mixed-precision pre-processing.
    
    # Input arguments:
    - A::Matrix : Symmetric matrix A ∈ ℝⁿˣⁿ.
    - Tl::Type : Low precision type. Used to compute the eigenvectors of A to form a preconditioner.
    - Th::Type : High precision type. Used to apply the preconditioner to A.

    # Output arguments: Λ, V, Params, [timePreconditioner timeApply timeJacobi timeElse]
    - Λ::Vector : Λ ∈ ℝⁿ = (λ₁,…,λₙ) of eigenvalues sorted in ascending order.
    - V::Matrix : V ∈ ℝⁿˣⁿ = (v₁,…,vₙ) of eigenvectors.
    - Params::Tuple : (no_rotation, no_sweep) = (number of rotations, number of sweeps).
    - [timePreconditioner timeApply timeJacobi timeElse] : Timing information for each stage.

"""
mp3_jacobi_eigen(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat}) where T<:AbstractFloat = mp3_jacobi_eigen!(copy(A), Tl, Th)

"""
    mp3_jacobi_eigen!(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat})
    
    Same as mp3_jacobi_eigen, but saves space by overwriting the input A, instead of creating a copy.
"""
function mp3_jacobi_eigen!(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat}) where T <: AbstractFloat 
    # Setup the high precision 
    Al = Symmetric(Tl.(A)) # symmetric for eigen

    # Timing for constructing the preconditioner 
    tmp = time()

    # Scale the matrix to avoid overflow 
    isscale1 = 0; 
    anrm1 = maximum(abs, A);
    eps1 = T.(Base.eps(Tl)/2); 
    safmin1 = T.(Base.floatmin(Tl));
    smlnum1 = safmin1 / eps1; 
    bignum1 = 1 / smlnum1; 
    rmin1 = sqrt( smlnum1 );
    rmax1 = sqrt( bignum1 );
    if ( anrm1 > 0 ) && ( anrm1 < rmin1 ) 
        isscale1 = 1; 
        sigma1 = rmin1 / anrm1;
    elseif ( anrm1 > rmax1 )
        isscale1 = 1; 
        sigma1 = rmax1 / anrm1;
    end
    if isscale1 == 1
        Aprime = sigma1 * copy(A); 
    else
        Aprime = copy(A); 
    end

    # Compute the low-precision eigenvectors 
    Vl = eigen!(Symmetric(Tl.(Aprime))).vectors

    # Orthogonalize the eigenvectors
    Vu = T.(Vl)
    Qu = Matrix(qr!(Vu).Q)

    # Store the time for computing the preconditioner
    timePreconditioner = time()-tmp

    # Timing for applying the preconditioner
    tmp = time()

    # Apply the preconditioner at high precision
    Ah = Th.(A)
    Qh = Th.(Qu)
    mul!(Ah, Qh', Ah * Qh)

    # Scale the preconditioned matrix such that it does not overflow when 
    # demoted to low precision. 
    isscale = 0; 
    anrm = maximum(abs, Ah); 
    eps2 = Th.(Base.eps(T)/2);
    safmin = Th.(Base.floatmin(T));
    smlnum = safmin / eps2; 
    bignum = 1 / smlnum; 
    rmin = sqrt( smlnum );
    rmax = sqrt( bignum );
    if ( anrm > 0 ) && ( anrm < rmin )
        isscale = 1;
        sigma = rmin / anrm;
    elseif ( anrm > rmax )
        isscale = 1;
        sigma = rmax / anrm;
    end 
    if isscale == 1 
        Ah = sigma * Ah; 
    end

    # Check the number of zeros to detect underflow 
    nz = count(x->x==0, Ah);

    # Post-process the preconditioned matrix to make it symmetric
    A = T.(Ah); 
    hermitianpart!(A)

    nz_after = count(x->x==0, A);
    if nz_after != nz 
        @warn "The preconditioner has caused underflow. The number of zeros has changed from $nz to $nz_after."
    end

    # Store the time for applying the preconditioner
    timeApply = time()-tmp 

    # Timing for applying the Jacobi algorithm
    tmp = time()

    # Compute the eigensystem of the preconditioned matrix 
    Λ, Vu, Params = _jacobi_eigen!(A, Vu)

    # Store the time for applying the Jacobi algorithm
    timeJacobi = time() - tmp

    # Timing for everything else 
    tmp = time()
    
    # Compute the final eigenvector matrix and sort by eigenvalues
    V = A
    mul!(V, Qu, Vu)
    LinearAlgebra.sorteig!(Λ, V)

    # Scale the eigenvalues back to the original scale
    if isscale == 1
        Λ = Λ / sigma; 
    end

    # Store the time for everything else
    timeElse = time() - tmp

    return Λ, V, Params, [timePreconditioner, timeApply, timeJacobi, timeElse]
end

export jacobi_eigen, jacobi_eigen!, off, mp2_jacobi_eigen, mp2_jacobi_eigen!, mp3_jacobi_eigen, mp3_jacobi_eigen!


# precompilation:
using GenericLinearAlgebra, Quadmath
A = randn(Float64, 4, 4); A = A'A
jacobi_eigen(A)
mp2_jacobi_eigen(A, Float32)
mp3_jacobi_eigen(A, Float32, Float128)

A = randn(Float32, 4, 4); A = A'A
jacobi_eigen(A)
mp2_jacobi_eigen(A, Float16)
mp3_jacobi_eigen(A, Float16, Float64)

end # module