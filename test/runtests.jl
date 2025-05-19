using JacobiEigen
using Test
using LinearAlgebra
using GenericLinearAlgebra

@testset "JacobiEigen.jl" begin
    # Matrix 1: Float64
    A64 = randn(Float64, 50, 50)
    A64 = A64' * A64

    Λ_ref64, V_ref64 = eigen(A64)
    Λ_jac64, V_jac64, _ = jacobi_eigen(A64)
    Λ_mp2_64, V_mp2_64, _ = mp2_jacobi_eigen(A64, Float32)
    Λ_mp3_64, V_mp3_64, _ = mp3_jacobi_eigen(A64, Float32, Float128)

    @test isapprox(Λ_jac64, Λ_ref64; atol=1e-10, rtol=1e-10)
    @test isapprox(Λ_mp2_64, Λ_ref64; atol=1e-8, rtol=1e-8)
    @test isapprox(Λ_mp3_64, Λ_ref64; atol=1e-10, rtol=1e-10)
    @test isapprox(V_jac64' * V_jac64, I, atol=1e-10)
    @test isapprox(V_mp2_64' * V_mp2_64, I, atol=1e-8)
    @test isapprox(V_mp3_64' * V_mp3_64, I, atol=1e-10)

    # Backward error tests for Float64
    @test norm(A64 - V_jac64 * Diagonal(Λ_jac64) * V_jac64') / norm(A64) < 1e-10
    @test norm(A64 - V_mp2_64 * Diagonal(Λ_mp2_64) * V_mp2_64') / norm(A64) < 1e-8
    @test norm(A64 - V_mp3_64 * Diagonal(Λ_mp3_64) * V_mp3_64') / norm(A64) < 1e-10

    # Matrix 2: Float32
    A32 = randn(Float32, 50, 50)
    A32 = A32' * A32

    Λ_ref32, V_ref32 = eigen(A32)
    Λ_jac32, V_jac32, _ = jacobi_eigen(A32)
    Λ_mp2_32, V_mp2_32, _ = mp2_jacobi_eigen(A32, Float16)
    Λ_mp3_32, V_mp3_32, _ = mp3_jacobi_eigen(A32, Float16, Float64)

    @test isapprox(Λ_jac32, Λ_ref32; atol=1e-5, rtol=1e-5)
    @test isapprox(Λ_mp2_32, Λ_ref32; atol=1e-3, rtol=1e-3)
    @test isapprox(Λ_mp3_32, Λ_ref32; atol=1e-5, rtol=1e-5)
    @test isapprox(V_jac32' * V_jac32, I, atol=1e-5)
    @test isapprox(V_mp2_32' * V_mp2_32, I, atol=1e-3)
    @test isapprox(V_mp3_32' * V_mp3_32, I, atol=1e-5)

    # Backward error tests for Float32
    @test norm(A32 - V_jac32 * Diagonal(Λ_jac32) * V_jac32') / norm(A32) < 1e-5
    @test norm(A32 - V_mp2_32 * Diagonal(Λ_mp2_32) * V_mp2_32') / norm(A32) < 1e-3
    @test norm(A32 - V_mp3_32 * Diagonal(Λ_mp3_32) * V_mp3_32') / norm(A32) < 1e-5
end