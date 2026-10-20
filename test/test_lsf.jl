using LinearAlgebra: I

@testset "LSF" begin
    @testset "FWHM ↔ σ" begin
        σ = fwhm_to_sigma_kms(2.5, 5000.0)
        @test σ ≈ C_KMS * 2.5 / (2sqrt(2log(2)) * 5000.0)
        @test σ ≈ 63.65 atol = 0.01
        @test sigma_kms_to_fwhm(σ, 5000.0) ≈ 2.5
        @test fwhm_to_sigma_kms(2.5, [4000.0, 5000.0]) ≈ [fwhm_to_sigma_kms(2.5, 4000.0), σ]
    end

    @testset "GaussianLSF" begin
        wave = [4000.0, 5000.0]
        l = GaussianLSF(wave, 2.5)
        @test l.sigma_kms ≈ fwhm_to_sigma_kms(2.5, wave)
        @test npix(l) == 2
        @test GaussianLSF([1, 2, 3]) isa GaussianLSF{Float64}
        @test_throws ArgumentError GaussianLSF([-1.0, 2.0])
        s = Spectrum(wave, ones(2), ones(2); lsf_fwhm=[2.5, 2.5])
        @test GaussianLSF(s).sigma_kms ≈ l.sigma_kms
        @test_throws ArgumentError GaussianLSF(Spectrum(wave, ones(2), ones(2)))
    end

    @testset "BandedLSF validation" begin
        @test_throws ArgumentError BandedLSF(ones(3, 5), [-1, 0])
        @test_throws ArgumentError BandedLSF(ones(2, 5), [0, 0])
        @test npix(BandedLSF(ones(3, 5), [-1, 0, 1])) == 5
    end

    @testset "dense and apply agree" begin
        R = BandedLSF(rand(3, 8), [-1, 0, 2])
        M = dense_matrix(R)
        x, X = rand(8), rand(8, 4)
        @test apply_banded(R, x) ≈ M * x
        @test apply_banded(R, X) ≈ M * X
        @test M[1, 1] == R.bands[2, 1] && M[1, 3] == R.bands[3, 1]
        @test_throws DimensionMismatch apply_banded(R, rand(5))
    end

    @testset "scipy dia convention" begin
        n, offs = 7, [1, 0, -1]
        A = zeros(n, n)
        for i in 1:n, o in offs
            1 <= i + o <= n && (A[i, i + o] = rand())
        end
        data = zeros(length(offs), n)       # data[d, c] = A[c - offs[d], c]
        for (d, o) in enumerate(offs), c in 1:n
            1 <= c - o <= n && (data[d, c] = A[c - o, c])
        end
        @test dense_matrix(banded_from_dia(data, offs)) ≈ A
    end

    @testset "normalize_rows" begin
        R = normalize_rows(BandedLSF(rand(3, 10) .+ 0.1, [-1, 0, 1]))
        @test sum(dense_matrix(R); dims=2) ≈ ones(10, 1)
        @test_throws ArgumentError normalize_rows(BandedLSF(zeros(1, 4), [0]))
    end

    @testset "banded_gaussian" begin
        R = banded_gaussian(fill(3.0, 200))
        @test sum(dense_matrix(R); dims=2) ≈ ones(200, 1)
        x = zeros(200); x[100] = 1.0                      # delta → kernel
        @test profile_variance(apply_banded(R, x)) ≈ 9 + 1 / 12 rtol = 1e-4
        @test apply_banded(banded_gaussian(zeros(10)), collect(1.0:10)) ≈ collect(1.0:10)
        @test apply_banded(R, ones(200)) ≈ ones(200)      # flux-preserving, even at edges
    end

    @testset "inst_sigma_kms" begin
        vs = 50.0
        R = banded_gaussian(fill(3.0, 200))
        expected = vs * sqrt(9 + 1 / 12)
        @test all(≈(expected; rtol=1e-4), inst_sigma_kms(R, vs)[30:170])
        g = GaussianLSF(fill(123.0, 5))
        @test inst_sigma_kms(g, vs) == g.sigma_kms
        @test to_banded(g, vs) isa BandedLSF
        @test to_banded(R, vs) === R
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

@testset "fwhm_to_sigma_kms and sigma_kms_to_fwhm (cases)" begin
    @testset "simple" begin
        @test fwhm_to_sigma_kms(2sqrt(2log(2)), C_KMS) ≈ 1.0
    end
    @testset "invalid input" begin
        @test_throws ArgumentError fwhm_to_sigma_kms(-1.0, 5000.0)
        @test_throws ArgumentError fwhm_to_sigma_kms(2.5, 0.0)
        @test_throws ArgumentError sigma_kms_to_fwhm(-10.0, 5000.0)
        @test_throws ArgumentError sigma_kms_to_fwhm(10.0, [5000.0, -1.0])
    end
    @testset "realistic" begin
        # MILES 2.5 Å across the optical: σ falls as 1/λ, and the round trip is exact
        wave = collect(3600.0:100.0:7400.0)
        σ = fwhm_to_sigma_kms(2.5, wave)
        @test σ[1] ≈ 88.41 atol = 0.01
        @test σ .* wave ≈ fill(σ[1] * wave[1], length(wave))
        @test sigma_kms_to_fwhm(σ, wave) ≈ fill(2.5, length(wave))
    end
    @testset "pathological" begin
        @test fwhm_to_sigma_kms(0.0, 5000.0) == 0.0           # perfect resolution
        @test fwhm_to_sigma_kms([1.0, 2.0], 5000.0) ≈ [1.0, 2.0] .* fwhm_to_sigma_kms(1.0, 5000.0)
    end
end

@testset "GaussianLSF (cases)" begin
    @testset "simple" begin
        @test GaussianLSF([10.0, 20.0]).sigma_kms == [10.0, 20.0]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError GaussianLSF([1.0, 2.0], [2.5, 2.5, 2.5])     # length mismatch
        @test_throws ArgumentError GaussianLSF([1.0, 2.0], -1.0)
        s = Spectrum([1.0, 2.0], ones(2), ones(2))
        @test_throws ArgumentError GaussianLSF(s, [1.0, 1.5, 2.0])               # no lsf_fwhm
    end
    @testset "realistic" begin
        # interpolate a varying spectrograph LSF onto a finer model grid
        wave = collect(4000.0:2.0:6000.0)
        s = Spectrum(wave, ones(length(wave)), ones(length(wave));
                     lsf_fwhm=collect(range(2.0, 3.0; length=length(wave))))
        mw = model_grid(wave; oversample=2)
        l = GaussianLSF(s, mw)
        @test npix(l) == length(mw)
        fwhm_back = sigma_kms_to_fwhm(l.sigma_kms, mw)
        inside = 4000.0 .<= mw .<= 6000.0          # outside, the LSF is clamped to the end values
        @test fwhm_back[inside] ≈ 2.0 .+ (mw[inside] .- 4000.0) ./ 2000.0 rtol = 1e-6
        @test fwhm_back[.!inside] ≈ [mw_i < 4000 ? 2.0 : 3.0 for mw_i in mw[.!inside]]
    end
    @testset "pathological" begin
        # zero width everywhere is a valid "no instrument" LSF
        @test all(iszero, GaussianLSF(zeros(5)).sigma_kms)
        # grids beyond the spectrum are clamped to the end values
        s = Spectrum([5000.0, 5001.0], ones(2), ones(2); lsf_fwhm=[2.0, 3.0])
        l = GaussianLSF(s, [4000.0, 6000.0])
        @test sigma_kms_to_fwhm(l.sigma_kms, [4000.0, 6000.0]) ≈ [2.0, 3.0]
    end
end

@testset "BandedLSF and banded_from_dia (cases)" begin
    @testset "simple" begin
        R = BandedLSF(ones(1, 4), [0])
        @test dense_matrix(R) == I(4)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError BandedLSF(ones(2, 4), [0])
        @test_throws ArgumentError BandedLSF(ones(2, 4), [1, 1])
        @test_throws ArgumentError banded_from_dia(ones(3, 4), [1, 0])
    end
    @testset "realistic" begin
        # an 11-diagonal DESI-like resolution matrix, built from a known dense matrix
        n, half = 40, 5
        offs = collect(half:-1:-half)
        A = zeros(n, n)
        for i in 1:n, j in max(1, i - half):min(n, i + half)
            A[i, j] = exp(-(i - j - 0.2)^2 / (2 * 1.7^2))
        end
        A ./= sum(A; dims=2)
        data = zeros(length(offs), n)
        for (d, o) in enumerate(offs), c in 1:n
            if 1 <= c - o <= n
                data[d, c] = A[c - o, c]
            end
        end
        R = banded_from_dia(data, offs)
        @test dense_matrix(R) ≈ A
        @test Matrix(Specje.sparse_matrix(R)) ≈ A
        @test apply_banded(R, ones(n)) ≈ ones(n)
    end
    @testset "pathological" begin
        # offsets beyond the matrix size are ignored, leaving only in-range entries
        R = BandedLSF(ones(3, 3), [-5, 0, 5])
        @test dense_matrix(R) == I(3)
        # no diagonals at all: the zero matrix
        Z = BandedLSF(zeros(0, 4), Int[])
        @test apply_banded(Z, ones(4)) == zeros(4)
    end
end

@testset "normalize_rows and apply_banded (cases)" begin
    @testset "simple" begin
        R = normalize_rows(BandedLSF(fill(2.0, 1, 3), [0]))
        @test dense_matrix(R) == I(3)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError normalize_rows(BandedLSF([1.0 0.0 1.0], [0]))   # empty row 2
        @test_throws DimensionMismatch apply_banded(BandedLSF(ones(1, 3), [0]), ones(4, 2))
    end
    @testset "realistic" begin
        rng = Xoshiro(2)
        R = normalize_rows(BandedLSF(rand(rng, 5, 50) .+ 0.1, collect(-2:2)))
        X = randn(rng, 50, 3)
        @test apply_banded(R, X) ≈ dense_matrix(R) * X
        @test apply_banded(R, X[:, 2]) ≈ dense_matrix(R) * X[:, 2]
    end
    @testset "pathological" begin
        # a single pixel: only the zero offset is in range
        R = normalize_rows(BandedLSF([1.0; 3.0; 1.0;;], [-1, 0, 1]))
        @test dense_matrix(R) == ones(1, 1)
        # apply_banded promotes to the input's element type (e.g. for AD numbers)
        @test eltype(apply_banded(R, BigFloat[2])) == BigFloat
    end
end

@testset "banded_gaussian and inst_sigma_kms (cases)" begin
    @testset "simple" begin
        R = banded_gaussian(fill(2.0, 100))
        @test all(≈(50.0 * sqrt(4 + 1 / 12); rtol=1e-4), inst_sigma_kms(R, 50.0)[20:80])
    end
    @testset "invalid input" begin
        @test_throws ArgumentError banded_gaussian([1.0, -1.0])
        @test_throws ArgumentError banded_gaussian([1.0, 1.0]; nsig=0)
        @test_throws ArgumentError inst_sigma_kms(banded_gaussian(ones(5)), 0.0)
        @test_throws ArgumentError to_banded(GaussianLSF(ones(5)), -1.0)
    end
    @testset "realistic" begin
        # a resolution that varies along the spectrum is reproduced row by row
        σ_pix = collect(range(1.5, 3.5; length=300))
        R = banded_gaussian(σ_pix)
        measured = inst_sigma_kms(R, 1.0)
        @test measured[30:270] ≈ sqrt.(σ_pix[30:270] .^ 2 .+ 1 / 12) rtol = 1e-3
        # GaussianLSF → banded → effective σ round trip
        g = GaussianLSF(fill(120.0, 300))
        @test inst_sigma_kms(to_banded(g, 40.0), 40.0)[150] ≈ sqrt(120.0^2 + 40.0^2 / 12) rtol = 1e-4
    end
    @testset "pathological" begin
        # zero width: the identity, whose effective σ is zero
        Z = banded_gaussian(zeros(10))
        @test dense_matrix(Z) == I(10)
        @test all(iszero, inst_sigma_kms(Z, 50.0))
        # a kernel much wider than the spectrum: rows are truncated but still sum to one
        W = banded_gaussian(fill(50.0, 20))
        @test sum(dense_matrix(W); dims=2) ≈ ones(20, 1)
        @test apply_banded(W, ones(20)) ≈ ones(20)
    end
end
