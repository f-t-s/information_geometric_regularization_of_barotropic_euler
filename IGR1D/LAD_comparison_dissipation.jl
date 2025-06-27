include("IGR1D.jl")
using DelimitedFiles

L = 1.0 
m = 500
T = 1.0
Δt_record = 0.0001
verbose = false

k = 40
β = 0.001
c = 0.0
s = 0.5
u0_funct, ρ0_funct = sine_wave(β, c, s, L, k)
xs, Δx, u0, ρ0 = initialize(m, L, u0_funct, ρ0_funct)

# Determining the gas law
a = 1.0
γ = 1.4
ν = 0e-3
Δt_igr = Δx / 1.2
Δt_lad = Δx / 1.2
Δt_rlw = Δx / 1.2

# small alpha
α_lad = 2.5 * Δx^2
α_igr= 2.5 * Δx^2
sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α_igr, Δx, Δt_igr, m)
sd_lad = SemidiscretizationRLWLAD(a, γ, ν, α_lad, Δx, Δt_lad, m)
sd_rlw = SemidiscretizationRLW(a, γ, ν, Δx, Δt_lad, m)

@time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_igr, T, ρ0, u0, Δt_record, verbose);
@time u_lad, ρ_lad, ts_lad = forward_euler(sd_lad, Δt_lad, T, ρ0, u0, Δt_record, verbose);
@time u_rlw, ρ_rlw, ts_rlw = forward_euler(sd_rlw, Δt_lad, T, ρ0, u0, Δt_record, verbose);

ekin_igr, epot_igr, equib_igr, etot_igr = compute_energies(sd_igr, u_igr, ρ_igr)
ekin_lad, epot_lad, equib_lad, etot_lad = compute_energies(sd_lad, u_lad, ρ_lad)
ekin_rlw, epot_rlw, equib_rlw, etot_rlw = compute_energies(sd_rlw, u_rlw, ρ_rlw)

base_path = "out/csv/lad_comparison_dissipation/"

open(base_path * "igr_low_alpha.csv", "w") do io
    writedlm(io, [etot_igr .- equib_igr ts_igr])
end

open(base_path * "lad_low_alpha.csv", "w") do io
    writedlm(io, [etot_lad .- equib_lad ts_lad])
end

open(base_path * "rlw.csv", "w") do io
    writedlm(io, [etot_rlw .- equib_rlw ts_rlw])
end

# large alpha
α_lad = 250 * Δx^2
α_igr= 250 * Δx^2
sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α_igr, Δx, Δt_igr, m)
sd_lad = SemidiscretizationRLWLAD(a, γ, ν, α_lad, Δx, Δt_lad, m)
sd_rlw = SemidiscretizationRLW(a, γ, ν, Δx, Δt_lad, m)

@time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_igr, T, ρ0, u0, Δt_record, verbose);
@time u_lad, ρ_lad, ts_lad = forward_euler(sd_lad, Δt_lad, T, ρ0, u0, Δt_record, verbose);
@time u_rlw, ρ_rlw, ts_rlw = forward_euler(sd_rlw, Δt_lad, T, ρ0, u0, Δt_record, verbose);

ekin_igr, epot_igr, equib_igr, etot_igr = compute_energies(sd_igr, u_igr, ρ_igr)
ekin_lad, epot_lad, equib_lad, etot_lad = compute_energies(sd_lad, u_lad, ρ_lad)
ekin_rlw, epot_rlw, equib_rlw, etot_rlw = compute_energies(sd_rlw, u_rlw, ρ_rlw)

base_path = "out/csv/lad_comparison_dissipation/"

open(base_path * "igr_high_alpha.csv", "w") do io
    writedlm(io, [etot_igr .- equib_igr ts_igr])
end

open(base_path * "lad_high_alpha.csv", "w") do io
    writedlm(io, [etot_lad .- equib_lad ts_lad])
end

open(base_path * "init_u.csv", "w") do io
    writedlm(io, [u0 xs])
end
