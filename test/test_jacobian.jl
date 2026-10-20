using LinearAlgebra
import Specje: log_jacobian, from_free, to_free

"Numerical log|det ∂f/∂u| by central differences."
function fd_logabsdet(f, u; h=1e-6)
    n = length(u)
    J = zeros(n, n)
    for j in 1:n
        e = zeros(n); e[j] = h
        J[:, j] = (f(u .+ e) .- f(u .- e)) ./ (2h)
    end
    return first(logabsdet(J))
end

@testset "change of variables" begin
    s = synthetic_setup()
    nd = length(s.grid.crop)
    star = StellarComponent(s.grid, s.T; σ_min=25.0)
    lsf = GaussianLSF(fill(60.0, nd))
    gas = GasComponent(s.dwave, [GasGroup("A", Multiplet("a", 4260.0)),
                                 GasGroup("B", Multiplet("b", 4400.0))], lsf; σ_min=10.0)
    poly = PolynomialComponent(nd, 1)

    @testset "stellar component" begin
        u = [1.3, -0.2]
        @test log_jacobian(star, u) ≈ fd_logabsdet(x -> from_free(star, x), u) atol = 1e-6
    end

    @testset "gas component" begin
        u = to_free(gas, [10.0, 80.0, -20.0, 120.0])
        @test log_jacobian(gas, u) ≈ fd_logabsdet(x -> from_free(gas, x), u) atol = 1e-6
    end

    @testset "polynomial has no parameters" begin
        @test log_jacobian(poly, Float64[]) == 0.0
    end

    @testset "full model" begin
        model = SpectralModel(s.grid, star, gas, poly)
        u = to_free(model, [120.0, 200.0, 10.0, 80.0, -20.0, 120.0])
        @test log_jacobian(model, u) ≈ fd_logabsdet(x -> from_free(model, x), u) atol = 1e-6
    end

    @testset "joint model with and without the calibration scale" begin
        f = joint_fixture(fit_scale=true)
        u = to_free(f.jm, [120.0, 200.0, 1.3])
        @test log_jacobian(f.jm, u) ≈ fd_logabsdet(x -> from_free(f.jm, x), u) atol = 1e-6
        jm0 = JointModel(f.stars, f.phot)
        u0 = to_free(jm0, [120.0, 200.0])
        @test log_jacobian(jm0, u0) ≈ fd_logabsdet(x -> from_free(jm0, x), u0) atol = 1e-6
    end
end
