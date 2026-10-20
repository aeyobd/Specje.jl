using LinearAlgebra: I
using SparseArrays: nnz, nonzeros

@testset "rebin" begin
    e = collect(0.0:4.0)
    f = [1.0, 2.0, 3.0, 4.0]

    @testset "rebin_flux" begin
        @test rebin_flux([0.0, 2.0, 4.0], e, f) ≈ [1.5, 3.5]
        @test rebin_flux([0.5, 1.5], e, f) ≈ [1.5]
        @test isnan(only(rebin_flux([-1.0, 1.0], e, f)))
        @test isnan(only(rebin_flux([3.0, 5.0], e, f)))
    end

    @testset "flux conservation" begin
        ne = [0.3, 1.7, 3.1]
        @test sum(rebin_flux(ne, e, f) .* diff(ne)) ≈ 6.1
    end

    @testset "matrix input" begin
        F = [f 2f]
        R = rebin_flux([0.0, 2.0, 4.0], e, F)
        @test size(R) == (2, 2)
        @test R[:, 2] ≈ 2 .* R[:, 1]
    end

    @testset "rebin_noise" begin
        @test rebin_noise([0.0, 2.0, 4.0], e, fill(0.3, 4)) ≈ fill(0.3 / sqrt(2), 2)
        @test rebin_noise([0.0, 1.0], e, fill(0.3, 4)) ≈ [0.3]
    end

    @testset "rebin_good" begin
        @test rebin_good([0.0, 2.0, 4.0], e, [true, false, true, true]) == [false, true]
        @test rebin_good([-1.0, 1.0], e, trues(4)) == [false]
    end

    @testset "rebin_log" begin
        w = collect(4000.0:1.0:5000.0)
        s = Spectrum(w, fill(2.0, length(w)), fill(0.1, length(w)))
        out = rebin_log(s; velscale=100.0)
        @test is_loguniform(out)
        @test velscale(out) ≈ 100.0 rtol = 1e-8
        @test all(out.good)
        @test all(≈(2.0), out.flux)
        @test all(<=(0.1 + 1e-12), out.noise)
        @test first(out.wave) >= first(w) && last(out.wave) <= last(w)

        ramp = Spectrum(w, w, fill(0.1, length(w)))
        @test rebin_log(ramp; velscale=150.0).flux ≈ rebin_log(ramp; velscale=150.0).wave rtol = 1e-3
    end

    @testset "rebin_log masks and LSF" begin
        w = collect(4000.0:1.0:5000.0)
        g = trues(length(w)); g[500] = false
        s = Spectrum(w, ones(length(w)), fill(0.1, length(w));
                     good=g, lsf_fwhm=fill(2.5, length(w)))
        out = rebin_log(s; velscale=100.0)
        @test count(!, out.good) >= 1
        @test all(out.flux[.!out.good] .== 0)
        @test all(≈(2.5), out.lsf_fwhm)
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "rebin_flux (cases)" begin
    e = collect(0.0:4.0)
    f = [1.0, 2.0, 3.0, 4.0]
    @testset "simple" begin
        @test rebin_flux(e, e, f) ≈ f                       # same edges: identity
    end
    @testset "invalid input" begin
        @test_throws DimensionMismatch rebin_flux(e, e, f[1:3])
        @test_throws ArgumentError rebin_flux([2.0, 1.0], e, f)
        @test_throws ArgumentError rebin_flux([1.0], e, f)
    end
    @testset "realistic" begin
        # SSP-like spectrum at 0.5 Å rebinned to 1.3 Å bins: integrated flux is conserved
        old = collect(4000.0:0.5:5000.0)
        flux = @. 1 + 0.3 * sin(old / 7) - 0.5 * exp(-(old - 4500)^2 / 8)
        new = collect(4000.0:1.3:4999.0)
        old_edges = pixel_edges(old)
        out = rebin_flux(new, old_edges, flux)
        # exact integral of the old piecewise-constant spectrum over [new[1], new[end]]
        full = findall(i -> old_edges[i] >= new[1] && old_edges[i+1] <= new[end], 1:length(old))
        partial_lo = (old_edges[full[1]] - new[1]) * flux[full[1] - 1]
        partial_hi = (new[end] - old_edges[full[end] + 1]) * flux[full[end] + 1]
        expected = sum(flux[full] .* 0.5) + partial_lo + partial_hi
        @test sum(out .* diff(new)) ≈ expected rtol = 1e-12
    end
    @testset "pathological" begin
        @test rebin_flux([1.2, 1.7], e, f) ≈ [2.0]           # new bin inside one old bin
        @test isnan(only(rebin_flux([-1.0, 5.0], e, f)))     # wider than the old range
        @test rebin_flux(e, e, reshape(f, :, 1)) ≈ reshape(f, :, 1)   # one-column matrix
        @test all(isnan, rebin_flux([10.0, 11.0, 12.0], e, f))        # disjoint ranges
    end
end

@testset "rebin_noise (cases)" begin
    e = collect(0.0:4.0)
    @testset "simple" begin
        @test rebin_noise(e, e, fill(0.2, 4)) ≈ fill(0.2, 4)
    end
    @testset "invalid input" begin
        @test_throws DimensionMismatch rebin_noise(e, e, fill(0.2, 3))
    end
    @testset "realistic" begin
        # the propagated error matches the Monte Carlo scatter of rebinned noise
        old = collect(0.0:0.5:100.0)
        new = collect(0.0:1.7:99.0)
        σ = fill(0.3, length(old) - 1)
        rng = Xoshiro(3)
        draws = reduce(hcat, [rebin_flux(new, old, 0.3 .* randn(rng, length(σ))) for _ in 1:4000])
        @test vec(std(draws; dims=2)) ≈ rebin_noise(new, old, σ) rtol = 0.06
    end
    @testset "pathological" begin
        @test rebin_noise(e, e, zeros(4)) == zeros(4)
        @test isnan(only(rebin_noise([-1.0, 1.0], e, ones(4))))
    end
end

@testset "rebin_good (cases)" begin
    e = collect(0.0:4.0)
    @testset "simple" begin
        @test rebin_good(e, e, [true, false, true, true]) == [true, false, true, true]
    end
    @testset "invalid input" begin
        @test_throws DimensionMismatch rebin_good(e, e, trues(3))
    end
    @testset "realistic" begin
        # one bad old pixel spoils exactly the new pixels that overlap it
        old = collect(0.0:1.0:20.0)
        good = trues(20); good[11] = false                  # covers [10, 11]
        new = collect(0.0:2.5:20.0)
        @test rebin_good(new, old, good) == [true, true, true, true, false, true, true, true]
    end
    @testset "pathological" begin
        @test !any(rebin_good(e, e, falses(4)))
        # a new edge exactly on a bad pixel's edge does not touch it
        @test rebin_good([0.0, 1.0], e, [true, false, false, false]) == [true]
    end
end

@testset "rebin_log (cases)" begin
    @testset "simple" begin
        w = collect(4000.0:1.0:4100.0)
        out = rebin_log(Spectrum(w, ones(101), fill(0.1, 101)); velscale=60.0)
        @test all(≈(1.0), out.flux)
    end
    @testset "invalid input" begin
        s = Spectrum(collect(4000.0:1.0:4100.0), ones(101), ones(101))
        @test_throws ArgumentError rebin_log(s; velscale=0.0)
        @test_throws ArgumentError rebin_log(s; velscale=-10.0)
    end
    @testset "realistic" begin
        # linear 0.5 Å grid with noise and an absorption line: integrated flux and line
        # equivalent width survive rebinning
        w = collect(4800.0:0.5:5200.0)
        flux = @. 1 - 0.6 * exp(-(w - 5000)^2 / (2 * 1.5^2))
        rng = Xoshiro(5)
        s = Spectrum(w, flux .+ 0.01 .* randn(rng, length(w)), fill(0.01, length(w)))
        out = rebin_log(s)
        ew(wv, fl) = sum((1 .- fl) .* diff(pixel_edges(wv)))
        @test ew(out.wave, out.flux) ≈ ew(w, s.flux) rtol = 0.02
        @test median(out.noise) < 0.01 * 1.01
    end
    @testset "pathological" begin
        w = collect(4000.0:1.0:4100.0)
        # every pixel bad: everything is masked, nothing is NaN
        bad = rebin_log(Spectrum(w, ones(101), ones(101); good=falses(101)); velscale=60.0)
        @test !any(bad.good) && all(==(0), bad.flux) && all(==(Inf), bad.noise)
        # a velscale wider than the whole spectrum leaves at most one pixel
        # 4000–4100 Å spans ≈ 7400 km/s: a log grid needs at least two pixels, so a
        # 5000 km/s or 10⁴ km/s velscale is rejected with a clear message
        @test npix(rebin_log(Spectrum(w, ones(101), ones(101)); velscale=3000.0)) == 2
        @test_throws ArgumentError rebin_log(Spectrum(w, ones(101), ones(101)); velscale=5000.0)
        @test_throws ArgumentError rebin_log(Spectrum(w, ones(101), ones(101)); velscale=1e4)
    end
end

@testset "model_grid (cases)" begin
    @testset "simple" begin
        g = model_grid([1000.0, 1001.0, 1002.0]; oversample=1)
        @test is_loguniform(g)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError model_grid([1.0, 2.0, 3.0]; oversample=0.5)
        @test_throws ArgumentError model_grid([3.0, 2.0, 1.0])
        @test_throws ArgumentError model_grid([1.0])
    end
    @testset "realistic" begin
        # DESI-like 0.8 Å pixels: the model grid covers every pixel at ≥ 2× resolution
        native = collect(3600.0:0.8:5900.0)
        g = model_grid(native; oversample=2)
        e, ge = pixel_edges(native), log_pixel_edges(g)
        @test first(ge) <= first(e) && last(ge) >= last(e)
        narrowest = C_KMS * log(e[end] / e[end-1])          # the reddest pixel, in km/s
        @test velscale(g) ≈ narrowest / 2 rtol = 1e-9
        @test size(pixel_projection(g, native)) == (length(native), length(g))
    end
    @testset "pathological" begin
        # very irregular pixels: the narrowest one sets the resolution
        native = [5000.0, 5000.1, 5010.0, 5020.0]
        g = model_grid(native; oversample=1)
        @test velscale(g) ≈ C_KMS * minimum(diff(log.(pixel_edges(native)))) rtol = 1e-9
    end
end

@testset "pixel_projection (cases)" begin
    mw = loggrid(4000.0, 4100.0, 10.0)
    @testset "simple" begin
        P = pixel_projection(mw, mw)
        @test Matrix(P) ≈ I(length(mw))        # same (log-uniform) pixels: identity
    end
    @testset "invalid input" begin
        native = collect(4010.0:1.0:4090.0)
        @test_throws ArgumentError pixel_projection(mw, native; sampling=:nearest)
        @test_throws ArgumentError pixel_projection(mw, [3990.0; native])
        @test_throws ArgumentError pixel_projection(mw, [3990.0; native]; sampling=:point)
        @test_throws DimensionMismatch pixel_projection(mw, native; data_edges=collect(4000.0:1.0:4010.0))
        @test_throws ArgumentError pixel_projection([4000.0], native)
    end
    @testset "realistic" begin
        # a smooth model projected onto coarser linear pixels equals its pixel averages
        native = collect(4010.0:0.7:4090.0)
        P = pixel_projection(mw, native)
        model = @. 2 + sin(mw / 3)
        e = pixel_edges(native)
        exact = [(-3cos(e[i+1] / 3) + 3cos(e[i] / 3)) / (e[i+1] - e[i]) + 2 for i in 1:length(native)]
        @test P * model ≈ exact rtol = 2e-3
    end
    @testset "pathological" begin
        # a data pixel narrower than one model pixel takes that pixel's value
        P = pixel_projection(mw, [4050.0]; data_edges=[4050.0, 4050.01])
        @test nnz(P) == 1 && only(nonzeros(P)) ≈ 1.0
        # a point sample exactly on a model node has one non-zero weight; a single data
        # pixel is fine for point sampling, which needs no pixel edges
        Pp = pixel_projection(mw, [mw[5]]; sampling=:point)
        @test nnz(Pp) == 1
        @test_throws ArgumentError pixel_projection(mw, Float64[])
        # a single data pixel with average sampling has no edges to infer
        @test_throws ArgumentError pixel_projection(mw, [4050.0])
    end
end
