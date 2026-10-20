import Specje: columns, in_bounds, to_free, from_free

@testset "gas component" begin
    s = synthetic_setup()
    dwave, nd, vs = s.dwave, length(s.dwave), s.vs        # ≈ 4205–4495 Å
    lsf = GaussianLSF(fill(60.0, nd))
    single(wave) = GasComponent(dwave, [GasGroup("g", Multiplet("x", wave))], lsf)
    masses(c, col) = col .* c.widths                       # flux per pixel

    @testset "Multiplet and GasGroup" begin
        m = Multiplet("d", [5008.24, 4960.3], [2.98, 1.0])
        @test m.name == "d" && m.ratios == [2.98, 1.0]
        @test Multiplet("s", 6564.6).ratios == [1.0]
        @test_throws ArgumentError Multiplet("bad", [1.0, 2.0], [1.0])
        @test_throws ArgumentError Multiplet("bad", [1.0], [-1.0])
        @test_throws ArgumentError Multiplet("bad", [0.0], [1.0])
        @test_throws ArgumentError Multiplet("bad", Float64[], Float64[])
        @test length(GasGroup("g", Multiplet("a", 4300.0), Multiplet("b", 4400.0)).multiplets) == 2
    end

    @testset "default_gas_groups" begin
        g = default_gas_groups()
        names = [m.name for gr in g for m in gr.multiplets]
        @test allunique(names)
        @test length(g) == 3 && length(names) == 12
        oiii = only(filter(m -> m.name == "[OIII]", g[3].multiplets))
        @test oiii.ratios == [1.0, 2.98]
    end

    @testset "range selection" begin
        grp = GasGroup("in", Multiplet("a", 4300.0), Multiplet("far", 6563.0),
                       Multiplet("p", [4350.0, 6563.0], [1.0, 3.0]))
        c = GasComponent(dwave, [grp], lsf)
        @test line_names(c) == ["a", "p"]
        @test c.groups[1].multiplets[2].waves == [4350.0]
        @test c.groups[1].multiplets[2].ratios == [1.0]
        @test_throws ArgumentError GasComponent(dwave, [GasGroup("out", Multiplet("far", 6563.0))], lsf)
    end

    @testset "construction checks" begin
        grp = [GasGroup("g", Multiplet("x", 4350.0))]
        @test_throws ArgumentError GasComponent(dwave, grp, GaussianLSF(fill(60.0, 5)))
        @test_throws ArgumentError GasComponent(dwave, grp, lsf; σ_min=0.0)
    end

    @testset "bookkeeping" begin
        g1 = GasGroup("A", Multiplet("a", 4260.0), Multiplet("b", [4300.0, 4400.0], [1.0, 3.0]))
        g2 = GasGroup("B", Multiplet("c", 4470.0))
        c = GasComponent(dwave, [g1, g2], lsf)
        @test nparams(c) == 4 && ncolumns(c) == 3
        @test line_names(c) == ["a", "b", "c"] && group_names(c) == ["A", "B"]
        out = columns(c, s.grid, [0.0, 80.0, 10.0, 90.0])
        @test size(out) == (nd, 3) && all(>=(0), out)
        @test count(!iszero, out[:, 3]) < 100              # compact support
    end

    @testset "Gaussian profile: flux, centroid, width" begin
        c = single(4350.0)
        v, σ = 30.0, 100.0
        m = masses(c, columns(c, s.grid, [v, σ])[:, 1])
        @test sum(m) ≈ 1.0 rtol = 1e-9
        centroid = sum(log.(dwave) .* m) / sum(m)
        @test centroid ≈ log(4350.0) + v / C_KMS atol = 1e-3 * vs / C_KMS
        @test profile_variance(m) ≈ (hypot(σ, 60.0) / vs)^2 + 1 / 12 rtol = 1e-4
    end

    @testset "multiplet ratios" begin
        c = GasComponent(dwave, [GasGroup("g", Multiplet("d", [4300.0, 4400.0], [1.0, 3.0]))], lsf)
        m = masses(c, columns(c, s.grid, [0.0, 100.0])[:, 1])
        @test sum(m[dwave .< 4350]) ≈ 1.0 rtol = 1e-9
        @test sum(m[dwave .> 4350]) ≈ 3.0 rtol = 1e-9
    end

    @testset "velocity moves the line" begin
        c = single(4350.0)
        peak(v) = argmax(columns(c, s.grid, [v, 100.0])[:, 1])
        @test peak(10vs) - peak(0.0) == 10
        @test peak(-10vs) - peak(0.0) == -10
    end

    @testset "line off the grid edge is truncated, not an error" begin
        c = single(4208.0)
        m = masses(c, columns(c, s.grid, [-300.0, 100.0])[:, 1])
        @test 0 < sum(m) < 1
    end

    @testset "banded LSF is applied exactly" begin
        R = banded_gaussian(fill(60.0 / vs, nd))
        grp = [GasGroup("g", Multiplet("x", 4350.0))]
        c = GasComponent(dwave, grp, R)
        m = masses(c, columns(c, s.grid, [0.0, 100.0])[:, 1])
        @test sum(m) ≈ 1.0 rtol = 1e-4
        @test profile_variance(m) ≈ ((100.0 / vs)^2 + (60.0 / vs)^2 + 2 / 12) rtol = 1e-3
    end

    @testset "parameter maps and bounds" begin
        c = GasComponent(dwave, [GasGroup("A", Multiplet("a", 4260.0)),
                                 GasGroup("B", Multiplet("b", 4400.0))], lsf;
                         σ_min=10.0, σ_max=500.0, vmax=400.0)
        θ = [10.0, 80.0, -20.0, 120.0]
        @test from_free(c, to_free(c, θ)) ≈ θ
        @test to_free(c, θ)[1] ≈ 10.0 / vs
        @test all(>=(10.0), from_free(c, [0.0, -50.0, 0.0, -50.0])[[2, 4]])
        @test_throws ArgumentError to_free(c, [0.0, 5.0, 0.0, 80.0])
        @test in_bounds(c, θ)
        @test !in_bounds(c, [500.0, 80.0, 0.0, 80.0])
        @test !in_bounds(c, [0.0, 80.0, 0.0, 800.0])
        @test !in_bounds(c, [0.0, 80.0, 0.0, 5.0])
        @test !in_bounds(c, [NaN, 80.0, 0.0, 80.0])
    end

    @testset "joint stellar + gas fit" begin
        star = StellarComponent(s.grid, s.T; σ_min=25.0, vmax=1000.0)
        gas = GasComponent(dwave, [GasGroup("A", Multiplet("a", 4260.0), Multiplet("b", 4340.0))],
                           lsf; σ_min=10.0, vmax=500.0)
        model = SpectralModel(s.grid, star, gas)
        @test nparams(model) == 4 && ncolumns(model) == 8
        θtrue = [120.0, 200.0, 30.0, 100.0]
        wtrue = [0.0, 2.0, 0.0, 1.0, 0.5, 0.0, 3.0, 1.5]
        flux = design_matrix(model, θtrue) * wtrue
        spec = Spectrum(dwave, flux, fill(0.01 * mean(flux), nd))
        res = fit_kinematics(model, spec, [100.0, 150.0, 0.0, 70.0])
        @test res.θ[1] ≈ 120.0 atol = 3.0
        @test res.θ[2] ≈ 200.0 atol = 6.0
        @test res.θ[3] ≈ 30.0 atol = 3.0
        @test res.θ[4] ≈ 100.0 atol = 8.0
        @test res.weights[7:8] ≈ [3.0, 1.5] atol = 0.2
        @test split_weights(model, res.weights)[2] == res.weights[7:8]
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "Multiplet and GasGroup (cases)" begin
    @testset "simple" begin
        m = Multiplet("Hb", 4862.683)
        @test m.waves == [4862.683] && m.ratios == [1.0]
        @test GasGroup("g", m).multiplets == [m]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError Multiplet("x", -10.0)
        @test_throws ArgumentError Multiplet("x", [5000.0, 5010.0], [1.0, 0.0])
    end
    @testset "realistic" begin
        oiii = Multiplet("[OIII]", [4960.295, 5008.240], [1.0, 2.98])
        @test sum(oiii.ratios) ≈ 3.98
        g = only(filter(g -> g.name == "high-ion", default_gas_groups()))
        @test any(m -> m.name == "[OIII]" && m.ratios == oiii.ratios, g.multiplets)
    end
    @testset "pathological" begin
        # integer wavelengths and ratios are accepted and stored as Float64
        m = Multiplet("x", [5000, 5010], [1, 3])
        @test m.waves isa Vector{Float64} && m.ratios == [1.0, 3.0]
    end
end

@testset "default_gas_groups (cases)" begin
    groups = default_gas_groups()
    @testset "simple" begin
        @test [g.name for g in groups] == ["balmer", "low-ion", "high-ion"]
    end
    @testset "realistic" begin
        # every line is optical/near-UV and in vacuum (Hα vacuum = 6564.6 Å, air = 6562.8 Å)
        waves = [λ for g in groups for m in g.multiplets for λ in m.waves]
        @test all(3700 .< waves .< 6800)
        halpha = only(m for m in groups[1].multiplets if m.name == "Halpha")
        @test only(halpha.waves) ≈ 6564.6 atol = 0.1
    end
    @testset "pathological" begin
        # a fresh copy each call: editing one does not change the defaults
        g1 = default_gas_groups()
        push!(g1[1].multiplets, Multiplet("extra", 5000.0))
        @test length(default_gas_groups()[1].multiplets) == 4
    end
end

@testset "GasComponent (cases)" begin
    s = synthetic_setup()
    dwave, nd = s.dwave, length(s.dwave)                    # ≈ 4205–4495 Å
    lsf = GaussianLSF(fill(60.0, nd))
    widths = GasComponent(dwave, [GasGroup("g", Multiplet("x", 4350.0))]).widths
    flux_of(col) = sum(col .* widths)
    @testset "simple" begin
        c = GasComponent(dwave, [GasGroup("g", Multiplet("x", 4350.0))], lsf)
        @test flux_of(columns(c, s.grid, [0.0, 50.0])[:, 1]) ≈ 1.0 rtol = 1e-9
    end
    @testset "invalid input" begin
        grp = [GasGroup("g", Multiplet("x", 4350.0))]
        @test_throws ArgumentError GasComponent(dwave, GasGroup[], lsf)
        @test_throws ArgumentError GasComponent(dwave, grp, lsf; σ_min=50.0, σ_max=40.0)
        @test_throws ArgumentError GasComponent(dwave, grp, lsf; vmax=0.0)
        @test_throws ArgumentError GasComponent(dwave, grp, GaussianLSF(fill(60.0, nd - 1)))
        c = GasComponent(dwave, grp, lsf)
        @test_throws ArgumentError columns(c, s.grid, [0.0])
        @test_throws ArgumentError to_free(c, [0.0, 0.5])                    # σ below σ_min
    end
    @testset "realistic" begin
        # an Hγ + [OIII]-like doublet complex: fluxes recovered by least squares in noise
        groups = [GasGroup("balmer", Multiplet("Hg", 4341.692)),
                  GasGroup("forbidden", Multiplet("d", [4363.2, 4300.0], [1.0, 0.5]))]
        c = GasComponent(dwave, groups, lsf; σ_min=5.0)
        θ = [30.0, 90.0, -20.0, 60.0]
        A = columns(c, s.grid, θ)
        truth = [4.0, 1.5]
        rng = Xoshiro(41)
        σn = 0.01
        data = Spectrum(dwave, A * truth .+ σn .* randn(rng, nd), fill(σn, nd))
        w, chi2 = solve_weights(A, FitData(data))
        @test w ≈ truth rtol = 0.02
        @test flux_of(A[:, 2]) ≈ 1.5 rtol = 1e-6                    # 1 + 0.5 within the column
    end
    @testset "pathological" begin
        c = GasComponent(dwave, [GasGroup("g", Multiplet("x", 4350.0))]; σ_min=0.01)
        # a line far narrower than a pixel still carries unit flux (pixel-integrated)
        col = columns(c, s.grid, [0.0, 0.01])[:, 1]
        @test flux_of(col) ≈ 1.0 rtol = 1e-9
        @test count(!iszero, col) <= 2
        # a line broader than the whole spectrum keeps only the flux inside the grid
        μ, width = log(4350.0), 1e4 / C_KMS
        inside = Specje.gauss_cdf((last(c.lnedges) - μ) / width) - Specje.gauss_cdf((first(c.lnedges) - μ) / width)
        @test flux_of(columns(c, s.grid, [0.0, 1e4])[:, 1]) ≈ inside rtol = 1e-6
        @test inside < 0.7
        # the same line in two groups gives identical columns at identical kinematics
        twin = GasComponent(dwave, [GasGroup("a", Multiplet("x", 4350.0)),
                                    GasGroup("b", Multiplet("y", 4350.0))])
        A = columns(twin, s.grid, [10.0, 80.0, 10.0, 80.0])
        @test A[:, 1] == A[:, 2]
    end
end
