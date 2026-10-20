@testset "SSPLibrary" begin
    lib = toy_library()

    @testset "shapes and accessors" begin
        @test (nwave(lib), nage(lib), nmet(lib)) == (50, 4, 3)
        @test ntemplates(lib) == 12
        @test age_gyr(lib)[1] ≈ 0.1
        @test age_gyr(lib)[end] ≈ 10.0
        @test lib isa SSPLibrary{Float64}
    end

    @testset "validation" begin
        w, f, a, z, m = lib.wave, lib.flux, lib.log_age, lib.logzsol, lib.mass_frac
        @test_throws ArgumentError SSPLibrary(w, f[:, :, 1:2], a, z, m)
        @test_throws ArgumentError SSPLibrary(w, f, reverse(a), z, m)
        @test_throws ArgumentError SSPLibrary(w, f, a, z, m[1:2, :])
        @test_throws ArgumentError SSPLibrary(w, f, a, z, m; norm=zeros(size(m)))
    end

    @testset "subset_library" begin
        s = subset_library(lib; log_age=(8.5, 10.0), logzsol=(-0.5, 0.0))
        @test nage(s) == 3 && nmet(s) == 2
        @test s.flux == lib.flux[:, 2:4, 2:3]
        @test s.mass_frac == lib.mass_frac[2:4, 2:3]
        @test_throws ArgumentError subset_library(lib; log_age=(20.0, 21.0))
    end

    @testset "crop_wave" begin
        c = crop_wave(lib, 4300.0, 4700.0)
        @test first(c.wave) >= 4300.0 && last(c.wave) <= 4700.0
        @test size(c.flux, 1) == length(c.wave)
        @test_throws ArgumentError crop_wave(lib, 9000.0, 9500.0)
    end

    @testset "flat templates" begin
        T = templates(lib)
        @test size(T) == (50, 12)
        @test T[:, 1] == lib.flux[:, 1, 1]
        @test T[:, 2] == lib.flux[:, 2, 1]               # age fastest
        @test T[:, nage(lib) + 1] == lib.flux[:, 1, 2]   # then metallicity
        p = template_params(lib)
        @test p.log_age[1:4] == lib.log_age
        @test p.logzsol[5] == lib.logzsol[2]
        @test reshape_weights(lib, collect(1.0:12.0))[2, 3] == 10.0
    end

    @testset "normalize_window" begin
        n = normalize_window(lib; window=(4400.0, 4600.0))
        idx = searchsortedfirst(n.wave, 4400.0):searchsortedlast(n.wave, 4600.0)
        @test all(≈(1.0), vec(mean(templates(n)[idx, :]; dims=1)))
        @test n.flux == lib.flux
        @test_throws ArgumentError normalize_window(lib; window=(7000.0, 7100.0))
    end

    @testset "masses" begin
        n = normalize_window(lib; window=(4400.0, 4600.0))
        w = ones(nage(n), nmet(n))
        @test formed_mass(n, w) ≈ 1 ./ n.norm
        @test stellar_mass(n, w) ≈ n.mass_frac ./ n.norm
        @test_throws ArgumentError formed_mass(n, ones(2, 2))
    end
end

# ---------------------------------------------------------------- per-function cases
# simple / invalid input / realistic / pathological

"FSPS-shaped library: `na` ages × `nz` metallicities with absorption lines and colours."
function fsps_like_library(; na=8, nz=4, wave=collect(3600.0:1.0:7400.0))
    log_age = collect(range(7.0, 10.1; length=na))
    logzsol = collect(range(-1.5, 0.25; length=nz))
    flux = Array{Float64}(undef, length(wave), na, nz)
    for ia in 1:na, iz in 1:nz
        f = 1e-4 .* (wave ./ 5500) .^ (-3 + 3 * (ia - 1) / (na - 1) + 0.3iz)
        for λ in (3934.8, 4102.9, 4305.0, 4862.7, 5176.7, 6564.6)
            f .*= 1 .- 0.05 * (ia + iz) .* exp.(-(wave .- λ) .^ 2 ./ 8)
        end
        flux[:, ia, iz] = f
    end
    mass_frac = [1.0 - 0.06 * (ia - 1) for ia in 1:na, iz in 1:nz]
    return SSPLibrary(wave, flux, log_age, logzsol, mass_frac; lsf_fwhm=fill(2.5, length(wave)))
end

@testset "SSPLibrary (cases)" begin
    @testset "simple" begin
        lib = SSPLibrary([1.0, 2.0], ones(2, 1, 1), [9.0], [0.0], ones(1, 1))
        @test ntemplates(lib) == 1 && templates(lib) == ones(2, 1)
    end
    @testset "invalid input" begin
        w, f = [1.0, 2.0, 3.0], ones(3, 2, 2)
        a, z, m = [8.0, 9.0], [-1.0, 0.0], ones(2, 2)
        @test_throws ArgumentError SSPLibrary(w, f, [9.0, 9.0], z, m)          # repeated age
        @test_throws ArgumentError SSPLibrary(w, f, a, z, m; norm=-ones(2, 2))
        @test_throws ArgumentError SSPLibrary(w, f, a, z, ones(2, 3))
        @test_throws ArgumentError SSPLibrary([3.0, 2.0, 1.0], f, a, z, m)
        bad = copy(f); bad[2, 1, 1] = NaN
        @test_throws ArgumentError SSPLibrary(w, bad, a, z, m)                 # non-finite flux
    end
    @testset "realistic" begin
        lib = fsps_like_library()
        @test (nage(lib), nmet(lib), ntemplates(lib)) == (8, 4, 32)
        @test age_gyr(lib) ≈ 10 .^ (lib.log_age .- 9)
        @test lib.lsf_fwhm == fill(2.5, nwave(lib))
    end
    @testset "pathological" begin
        # a library may contain zero flux (e.g. an SSP that is dark at some wavelengths)
        lib = SSPLibrary([1.0, 2.0], zeros(2, 1, 1), [9.0], [0.0], ones(1, 1))
        @test all(iszero, templates(lib))
        # integer inputs are promoted to floating point
        @test SSPLibrary([1, 2], ones(Int, 2, 1, 1), [9], [0], ones(Int, 1, 1)) isa SSPLibrary{Float64}
    end
end

@testset "subset_library (cases)" begin
    lib = fsps_like_library()
    @testset "simple" begin
        s = subset_library(lib; log_age=(lib.log_age[2], lib.log_age[3]))
        @test s.log_age == lib.log_age[2:3] && s.logzsol == lib.logzsol
    end
    @testset "invalid input" begin
        @test_throws ArgumentError subset_library(lib; log_age=(10.0, 8.0))
        @test_throws ArgumentError subset_library(lib; logzsol=(1.0, 2.0))     # empty
    end
    @testset "realistic" begin
        # young, metal-rich subset: flux, masses and norms stay aligned
        s = subset_library(lib; log_age=(-Inf, 9.0), logzsol=(-0.5, Inf))
        ia = findall(<=(9.0), lib.log_age)
        iz = findall(>=(-0.5), lib.logzsol)
        @test s.flux == lib.flux[:, ia, iz]
        @test s.mass_frac == lib.mass_frac[ia, iz] && s.norm == lib.norm[ia, iz]
    end
    @testset "pathological" begin
        # a zero-width range on a grid value keeps exactly that SSP
        s = subset_library(lib; log_age=(lib.log_age[4], lib.log_age[4]),
                           logzsol=(lib.logzsol[1], lib.logzsol[1]))
        @test ntemplates(s) == 1
        @test vec(templates(s)) ≈ lib.flux[:, 4, 1] ./ lib.norm[4, 1]
    end
end

@testset "crop_wave (cases)" begin
    lib = fsps_like_library()
    @testset "simple" begin
        @test crop_wave(lib, 4000.0, 4002.0).wave == [4000.0, 4001.0, 4002.0]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError crop_wave(lib, 5000.0, 4000.0)
        @test_throws ArgumentError crop_wave(lib, 100.0, 200.0)
    end
    @testset "realistic" begin
        c = crop_wave(lib, 4800.0, 5300.0)
        @test c.flux == lib.flux[1201:1701, :, :]
        @test c.lsf_fwhm == lib.lsf_fwhm[1201:1701]
    end
    @testset "pathological" begin
        @test nwave(crop_wave(lib, 5000.0, 5000.0)) == 1
        @test_throws ArgumentError crop_wave(lib, 5000.2, 5000.8)    # between pixels
    end
end

@testset "normalize_window (cases)" begin
    lib = fsps_like_library()
    @testset "simple" begin
        n = normalize_window(lib; window=(5300.0, 5700.0))
        @test n.norm ≈ dropdims(mean(lib.flux[1701:2101, :, :]; dims=1); dims=1)
    end
    @testset "invalid input" begin
        @test_throws ArgumentError normalize_window(lib; window=(5700.0, 5300.0))
        @test_throws ArgumentError normalize_window(lib; window=(5500.0, 5500.0))
        @test_throws ArgumentError normalize_window(lib; window=(9000.0, 9500.0))
    end
    @testset "realistic" begin
        # normalising twice gives the same result: norm is replaced, not compounded
        n1 = normalize_window(lib; window=(5300.0, 5700.0))
        n2 = normalize_window(n1; window=(5300.0, 5700.0))
        @test n2.norm == n1.norm
        window_means = vec(mean(templates(n1)[1701:2101, :]; dims=1))
        @test window_means ≈ ones(32)
    end
    @testset "pathological" begin
        # an SSP with no flux in the window cannot be normalised there
        f = ones(3, 2, 1); f[:, 2, 1] .= 0
        dark = SSPLibrary([1.0, 2.0, 3.0], f, [8.0, 9.0], [0.0], ones(2, 1))
        @test_throws ArgumentError normalize_window(dark; window=(1.0, 3.0))
        # a window narrower than a pixel but containing one centre uses that pixel
        n = normalize_window(lib; window=(5000.0, 5000.5))
        @test n.norm == lib.flux[1401, :, :]
    end
end

@testset "templates, template_params, reshape_weights (cases)" begin
    lib = normalize_window(fsps_like_library(); window=(5300.0, 5700.0))
    @testset "simple" begin
        w = zeros(ntemplates(lib)); w[1] = 1.0
        @test templates(lib) * w == lib.flux[:, 1, 1] ./ lib.norm[1, 1]
    end
    @testset "invalid input" begin
        @test_throws ArgumentError reshape_weights(lib, ones(5))
    end
    @testset "realistic" begin
        # a population mixture built on the (age, Z) grid is recovered column by column
        W = zeros(nage(lib), nmet(lib)); W[3, 2] = 0.7; W[7, 4] = 0.3
        mix = templates(lib) * vec(W)
        manual = 0.7 .* lib.flux[:, 3, 2] ./ lib.norm[3, 2] .+ 0.3 .* lib.flux[:, 7, 4] ./ lib.norm[7, 4]
        @test mix ≈ manual
        p = template_params(lib)
        k = findfirst(i -> p.log_age[i] == lib.log_age[7] && p.logzsol[i] == lib.logzsol[4], eachindex(p.log_age))
        @test vec(W)[k] == 0.3
        @test reshape_weights(lib, vec(W)) == W
    end
    @testset "pathological" begin
        one_ssp = subset_library(lib; log_age=(lib.log_age[1], lib.log_age[1]),
                                 logzsol=(lib.logzsol[1], lib.logzsol[1]))
        @test template_params(one_ssp) == (log_age=[lib.log_age[1]], logzsol=[lib.logzsol[1]])
        @test size(reshape_weights(one_ssp, [2.0])) == (1, 1)
    end
end

@testset "formed_mass and stellar_mass (cases)" begin
    lib = normalize_window(fsps_like_library(); window=(5300.0, 5700.0))
    @testset "simple" begin
        W = ones(nage(lib), nmet(lib))
        @test formed_mass(lib, W) ≈ 1 ./ lib.norm
    end
    @testset "invalid input" begin
        @test_throws ArgumentError formed_mass(lib, ones(3, 3))
        @test_throws ArgumentError stellar_mass(lib, ones(nmet(lib), nage(lib)))   # transposed
    end
    @testset "realistic" begin
        # 1 M☉ formed in one SSP gives back 1 M☉ formed and mass_frac M☉ surviving
        W = zeros(nage(lib), nmet(lib)); W[5, 3] = lib.norm[5, 3]
        @test sum(formed_mass(lib, W)) ≈ 1.0
        @test sum(stellar_mass(lib, W)) ≈ lib.mass_frac[5, 3]
        @test sum(stellar_mass(lib, W)) < sum(formed_mass(lib, W))
    end
    @testset "pathological" begin
        Z = zeros(nage(lib), nmet(lib))
        @test all(iszero, stellar_mass(lib, Z))
    end
end
