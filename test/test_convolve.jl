@testset "Fourier convolution" begin
    vs = 50.0
    ntemp = 4156
    twave = 3000.0 .* exp.((0:ntemp-1) .* vs ./ C_KMS)
    x0, s0 = 200.0, 4.0
    line = [exp(-(i - x0)^2 / (2 * s0^2)) for i in 0:ntemp-1]
    nd = 200

    data_grid(frac) = twave[101:300] .* exp(frac * vs / C_KMS)

    @testset "kinematics_grid geometry" begin
        kg = kinematics_grid(twave, data_grid(0.0))
        @test kg.crop == 101:300
        @test kg.frac ≈ 0.0 atol = 1e-9
        @test kg.nfft >= ntemp + ceil(Int, 2500 / vs)
        @test kg.velscale ≈ vs
        kg3 = kinematics_grid(twave, data_grid(0.3))
        @test kg3.crop == 101:300
        @test kg3.frac ≈ 0.3 atol = 1e-9
        @test shift_pixels(kg3, 100.0) ≈ 100.0 / vs - 0.3
    end

    @testset "grid validation" begin
        @test_throws ArgumentError kinematics_grid(twave, twave[1:200])          # no margin
        @test_throws ArgumentError kinematics_grid(twave, twave[end-199:end])
        @test_throws ArgumentError kinematics_grid(twave, collect(range(4000.0, 4100.0; length=200)))  # not log-uniform
        dwave60 = 4000.0 .* exp.((0:199) .* 60.0 ./ C_KMS)
        @test_throws ArgumentError kinematics_grid(twave, dwave60)
    end

    @testset "analytic Gaussian line" begin
        for frac in (0.0, 0.3, -0.4)
            kg = kinematics_grid(twave, data_grid(frac))
            Tft = transform_templates(kg, line)
            v_pix, σ_pix = 7.3, 3.0
            out = convolve_templates(kg, Tft, v_pix * vs, σ_pix * vs)
            s = sqrt(s0^2 + σ_pix^2)
            j = 0:nd-1
            expected = @. (s0 / s) * exp(-(kg.crop.start - 1 + j + kg.frac - x0 - v_pix)^2 / (2 * s^2))
            @test size(out) == (nd, 1)
            @test vec(out) ≈ expected atol = 1e-8
        end
    end

    @testset "linearity and columns" begin
        kg = kinematics_grid(twave, data_grid(0.0))
        Tft = transform_templates(kg, [line 2 .* line])
        out = convolve_templates(kg, Tft, 120.0, 90.0)
        @test out[:, 2] ≈ 2 .* out[:, 1]
    end

    @testset "negative velocity" begin
        kg = kinematics_grid(twave, data_grid(0.0))
        Tft = transform_templates(kg, line)
        up = convolve_templates(kg, Tft, 5vs, 3vs)
        dn = convolve_templates(kg, Tft, -5vs, 3vs)
        @test argmax(vec(up)) - argmax(vec(dn)) == 10
    end

    @testset "two components" begin
        kg = kinematics_grid(twave, data_grid(0.0))
        b1 = transform_templates(kg, [line 2 .* line])
        b2 = transform_templates(kg, line)
        out = convolve_blocks(kg, (b1, b2), ((0.0, 3vs), (200.0, 6vs)))
        @test size(out) == (nd, 3)
        @test out[:, 1:2] ≈ convolve_templates(kg, b1, 0.0, 3vs)
        @test out[:, 3] ≈ vec(convolve_templates(kg, b2, 200.0, 6vs))
    end

    @testset "template length guard" begin
        kg = kinematics_grid(twave, data_grid(0.0))
        @test_throws ArgumentError transform_templates(kg, ones(kg.nfft + 1))
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "kinematics_grid (cases)" begin
    vs = 50.0
    twave = 4000.0 .* exp.((0:999) .* vs ./ C_KMS)
    @testset "simple" begin
        kg = kinematics_grid(twave, twave[101:900])
        @test kg.crop == 101:900 && kg.velscale ≈ vs
    end
    @testset "invalid input" begin
        @test_throws ArgumentError kinematics_grid(twave, twave[101:900]; margin_kms=-1.0)
        @test_throws ArgumentError kinematics_grid(twave, twave[101:900]; margin_kms=1e4)
        @test_throws ArgumentError kinematics_grid(twave, twave[101:900] .* (1 .+ 1e-4 .* (1:800)))
    end
    @testset "realistic" begin
        # templates 3500–7500 Å at 30 km/s, data 4000–7000 Å offset by 0.37 px
        tw = loggrid(3500.0, 7500.0, 30.0)
        dw = loggrid(4000.0, 7000.0, 30.0) .* exp(0.37 * 30.0 / C_KMS)
        kg = kinematics_grid(tw, dw)
        @test length(kg.crop) == length(dw)
        @test abs(kg.frac) <= 0.5
        @test tw[first(kg.crop)] * exp(kg.frac * 30.0 / C_KMS) ≈ first(dw) rtol = 1e-12
        @test kg.nfft >= length(tw) + ceil(Int, 2500 / 30)
    end
    @testset "pathological" begin
        # no margin requested: the data may span the whole template grid
        kg = kinematics_grid(twave, twave; margin_kms=0.0)
        @test kg.crop == 1:1000
        # a data grid exactly half a pixel off rounds to one side with |frac| = ½
        kgh = kinematics_grid(twave, twave[101:900] .* exp(0.5 * vs / C_KMS))
        @test abs(kgh.frac) ≈ 0.5 atol = 1e-9
    end
end

@testset "transform_templates and convolve_templates (cases)" begin
    vs = 50.0
    twave = 4000.0 .* exp.((0:999) .* vs ./ C_KMS)
    kg = kinematics_grid(twave, twave[101:900])
    line = [exp(-(i - 500.0)^2 / (2 * 2.0^2)) for i in 0:999]
    @testset "simple" begin
        # v = 0, σ = 0 returns the template on the data pixels
        Tft = transform_templates(kg, line)
        @test vec(convolve_templates(kg, Tft, 0.0, 0.0)) ≈ line[101:900] atol = 1e-12
    end
    @testset "invalid input" begin
        other = kinematics_grid(twave[1:900], twave[101:800])
        Tft = transform_templates(other, line[1:900])
        @test_throws DimensionMismatch convolve_templates(kg, Tft, 0.0, 100.0)
        @test_throws ArgumentError transform_templates(kg, ones(kg.nfft + 5, 2))
    end
    @testset "realistic" begin
        # several SSP-like templates broadened by σ = 180 km/s keep their flux and line depth ratio
        T = 1 .- 0.3 .* [exp(-(i - c)^2 / (2 * 1.5^2)) for i in 0:999, c in (450.0, 500.0, 520.0)]
        Tft = transform_templates(kg, T)
        out = convolve_templates(kg, Tft, 75.0, 180.0)
        @test size(out) == (800, 3)
        @test vec(sum(1 .- out; dims=1)) ≈ vec(sum(1 .- T; dims=1)) rtol = 1e-8
    end
    @testset "pathological" begin
        Tft = transform_templates(kg, line)
        # σ below σ_templ: the kernel width is clamped to zero, not imaginary
        @test convolve_templates(kg, Tft, 0.0, 10.0; σ_templ=40.0) ≈ convolve_templates(kg, Tft, 0.0, 0.0)
        # σ exactly σ_templ: no extra broadening
        @test convolve_templates(kg, Tft, 0.0, 40.0; σ_templ=40.0) ≈ convolve_templates(kg, Tft, 0.0, 0.0)
        # an integer-pixel velocity with σ = 0 is an exact shift
        shifted = convolve_templates(kg, Tft, 3vs, 0.0)
        @test vec(shifted) ≈ line[98:897] atol = 1e-12
    end
end

@testset "convolve_blocks (cases)" begin
    vs = 50.0
    twave = 4000.0 .* exp.((0:999) .* vs ./ C_KMS)
    kg = kinematics_grid(twave, twave[101:900])
    line = [exp(-(i - 500.0)^2 / (2 * 2.0^2)) for i in 0:999]
    b = transform_templates(kg, line)
    @testset "simple" begin
        @test convolve_blocks(kg, (b,), ((0.0, 100.0),)) ≈ convolve_templates(kg, b, 0.0, 100.0)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError convolve_blocks(kg, (b, b), ((0.0, 100.0),))
    end
    @testset "realistic" begin
        # counter-rotating components: the two blocks peak on opposite sides
        out = convolve_blocks(kg, (b, b), ((-150.0, 80.0), (150.0, 80.0)))
        @test argmax(out[:, 2]) - argmax(out[:, 1]) == 6
    end
    @testset "pathological" begin
        # the same block twice with the same kinematics gives two identical columns
        out = convolve_blocks(kg, (b, b), ((0.0, 50.0), (0.0, 50.0)))
        @test out[:, 1] == out[:, 2]
    end
end
