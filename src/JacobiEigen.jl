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
    tol = sqrt(T(n)) * eps(T)/2
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
    TODO: Add explanation of Tl.
    
"""
mp2_jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}) = mp2_jacobi_eigen!(copy(A), Tl)

"""
    mp2_jacobi_eigen!(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat})
    
    Same as mp2_jacobi_eigen, but saves space by overwriting the input A, instead of creating a copy.
"""
function mp2_jacobi_eigen!(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}) where T <: AbstractFloat 

    # Compute the low-precision eigenvectors 
    Vl = eigen!(Symmetric(Tl.(A))).vectors

    # Orthogonalize the eigenvectors
    Vu = T.(Vl)
    Qu = Matrix(qr!(Vu).Q)

    # Apply the preconditioner
    mul!(Vu, A, Qu)
    mul!(A, Qu', Vu)
    
    # Post-process the preconditioned matrix to make it symmetric
    hermitianpart!(A)

    # Compute the eigensystem of the preconditioned matrix 
    Λ, Vu, Params = _jacobi_eigen!(A, Vu)

    # Compute the final eigenvector matrix and sort by eigenvalues
    V = A
    mul!(V, Qu, Vu)
    LinearAlgebra.sorteig!(Λ, V)

    return Λ, V, Params
end


"""
    mp3_jacobi_eigen(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat})

    Compute the spectral decomposition of a symmetric matrix A ∈ ℝⁿˣⁿ using the cyclic Jacobi algorithm with mixed-precision pre-processing.
    TODO: Add explanation of Tl, and Th.
    
"""
mp3_jacobi_eigen(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat}) where T<:AbstractFloat = mp3_jacobi_eigen!(copy(A), Tl, Th)

"""
    mp3_jacobi_eigen!(A::AbstractMatrix{<:AbstractFloat}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat})
    
    Same as mp3_jacobi_eigen, but saves space by overwriting the input A, instead of creating a copy.
"""
function mp3_jacobi_eigen!(A::AbstractMatrix{T}, Tl::Type{<:AbstractFloat}, Th::Type{<:AbstractFloat}) where T <: AbstractFloat 
    # Setup the high precision 
    Al = Symmetric(Tl.(A)) # symmetric for eigen

    # Compute the low-precision eigenvectors 
    Vl = eigen!(Al).vectors

    # Orthogonalize the eigenvectors
    Vu = T.(Vl)
    Qu = Matrix(qr!(Vu).Q)

    # Apply the preconditioner at high precision
    Ah = Th.(A)
    Qh = Th.(Qu)
    mul!(A, Qh', Ah * Qh)

    # Post-process the preconditioned matrix to make it symmetric
    hermitianpart!(A)

    # Compute the eigensystem of the preconditioned matrix 
    Λ, Vu, Params = _jacobi_eigen!(A, Vu)
    
    # Compute the final eigenvector matrix and sort by eigenvalues
    V = A
    mul!(V, Qu, Vu)
    LinearAlgebra.sorteig!(Λ, V)

    return Λ, V, Params
end


export jacobi_eigen, jacobi_eigen!, off, mp2_jacobi_eigen, mp2_jacobi_eigen!, mp3_jacobi_eigen, mp3_jacobi_eigen!


# precompilation:
using GenericLinearAlgebra, Quadmath
A = randn(Float64, 4, 4)
jacobi_eigen(A)
mp2_jacobi_eigen(A, Float32)
mp3_jacobi_eigen(A, Float32, Float128)
A = randn(Float32, 4, 4)
jacobi_eigen(A)
mp2_jacobi_eigen(A, Float16)
mp3_jacobi_eigen(A, Float16, Float64)

end # module