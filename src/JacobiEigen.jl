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
    # Initialize parameters 
    n = size( A,1 )
    tol = sqrt(T(n)) * eps(T)/2
    V = Matrix{T}(I,n,n)
    J = zeros(T,2,2)
    tmp1 = zeros(T,2,n)
    tmp2 = zeros(T,2,n)
    
    Params = _jacobi_eigen!(A, tol, V, J, tmp1, tmp2)

    # Post-processing 
    Λ = diag(A)
    sorted_indices = sortperm(Λ, rev = true)
    Λ = Λ[sorted_indices]
    V = V[:, sorted_indices]

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
                    # A[:,[p,q]] = A[:,[p,q]]*J'
                    @views tmp1[1,:] = A[p,:]
                    @views tmp1[2,:] = A[q,:]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views A[p,:] = tmp2[1,:]
                    @views A[q,:] = tmp2[2,:]

                    # A[[p,q],:] = J * A[[p,q],:]
                    @views tmp1[1,:] = A[:,p]
                    @views tmp1[2,:] = A[:,q]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views A[:,p] = tmp2[1,:]
                    @views A[:,q] = tmp2[2,:]
                    
                    A[p,q] = 0
                    A[q,p] = 0
                    
                    # V[p,q],:] = J * V[[p,q],:]
                    @views tmp1[1,:] = V[:,p]
                    @views tmp1[2,:] = V[:,q]
                    mul!(tmp2, J, tmp1) # tmp2 = J*tmp1
                    @views V[:,p] = tmp2[1,:]
                    @views V[:,q] = tmp2[2,:]
                end
            end
        end
    end
    return no_rotation, no_sweep
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

export jacobi_eigen, jacobi_eigen!, off

end # module