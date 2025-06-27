include("IGR1D.jl")
using DelimitedFiles

# problem size
L = 1.0
m = 500
T = 4.0

# initial condition
β = 3.0
c = 0.0
s = 0.0
u0_funct, ρ0_funct = sine_wave(β, c, s, L)
xs, Δx, u0, ρ0 = initialize(m, L, u0_funct, ρ0_funct)

# Determining the gas law
a = 1.0
γ = 1.4
ν = 0e-3
Δt_lf = Δx / 4.0
Δt_rlw = Δx / 4.0
Δt_igr = Δx / 4.0

α = 20 / m ^ 2

# Setting time step and running computation
Δt_record = 0.0001
# Whether to report the times at which snapshots are recorded
verbose = false
stype = Float16

sd_rlw = SemidiscretizationRLW(a, γ, ν, Δx, Δt_rlw, m)
sd_lf = SemidiscretizationLF(a, γ, ν, Δx, Δt_lf, m)
sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α, Δx, Δt_igr, m)

@time u_lf, ρ_lf, ts_lf = forward_euler(sd_lf, Δt_lf, T, ρ0, u0, Δt_record, verbose)
@time u_rlw, ρ_rlw, ts_rlw = forward_euler(sd_rlw, Δt_rlw, T, ρ0, u0, Δt_record, verbose);
@time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_igr, T, ρ0, u0, Δt_record, verbose);

# Reference solution: 
m_ref = 20000
xs_ref, Δx_ref, u0_ref, ρ0_ref = initialize(m_ref, L, u0_funct, ρ0_funct)
Δt_ref = Δx_ref / 4.25
sd_ref = SemidiscretizationLF(a, γ, ν, Δx_ref, Δt_ref, m_ref)
@time u_ref, ρ_ref, ts_ref = forward_euler(sd_ref, Δt_ref, T, ρ0_ref, u0_ref, Δt_record, verbose)


Ts = [0.0875, 0.75, 2.32, 4.0]
tsteps_lf = [findfirst(k -> ts_lf[k] ≥ T, eachindex(ts_lf)) for T in Ts]
tsteps_rlw = [findfirst(k -> ts_rlw[k] ≥ T, eachindex(ts_rlw)) for T in Ts]
tsteps_igr = [findfirst(k -> ts_igr[k] ≥ T, eachindex(ts_igr)) for T in Ts]
tsteps_ref = [findfirst(k -> ts_ref[k] ≥ T, eachindex(ts_ref)) for T in Ts]

base_path = "out/csv/1d_sine/"

open(base_path * "lf_T1.csv", "w") do io
    writedlm(io, [u_lf[:, tsteps_lf[1]] ρ_lf[:, tsteps_lf[1]] xs])
end

open(base_path * "lf_T2.csv", "w") do io
    writedlm(io, [u_lf[:, tsteps_lf[2]] ρ_lf[:, tsteps_lf[2]] xs])
end

open(base_path * "lf_T3.csv", "w") do io
    writedlm(io, [u_lf[:, tsteps_lf[3]] ρ_lf[:, tsteps_lf[3]] xs])
end

open(base_path * "lf_T4.csv", "w") do io
    writedlm(io, [u_lf[:, tsteps_lf[4]] ρ_lf[:, tsteps_lf[4]] xs])
end

open(base_path * "rlw_T1.csv", "w") do io
    writedlm(io, [u_rlw[:, tsteps_rlw[1]] ρ_rlw[:, tsteps_lf[1]] xs])
end

open(base_path * "rlw_T2.csv", "w") do io
    writedlm(io, [u_rlw[:, tsteps_rlw[2]] ρ_rlw[:, tsteps_rlw[2]] xs])
end

open(base_path * "rlw_T3.csv", "w") do io
    writedlm(io, [u_rlw[:, tsteps_rlw[3]] ρ_rlw[:, tsteps_rlw[3]] xs])
end

open(base_path * "rlw_T4.csv", "w") do io
    writedlm(io, [u_rlw[:, tsteps_rlw[4]] ρ_rlw[:, tsteps_rlw[4]] xs])
end

open(base_path * "igr_T1.csv", "w") do io
    writedlm(io, [u_igr[:, tsteps_igr[1]] ρ_igr[:, tsteps_igr[1]] xs])
end

open(base_path * "igr_T2.csv", "w") do io
    writedlm(io, [u_igr[:, tsteps_igr[2]] ρ_igr[:, tsteps_igr[2]] xs])
end

open(base_path * "igr_T3.csv", "w") do io
    writedlm(io, [u_igr[:, tsteps_igr[3]] ρ_igr[:, tsteps_igr[3]] xs])
end

open(base_path * "igr_T4.csv", "w") do io
    writedlm(io, [u_igr[:, tsteps_igr[4]] ρ_igr[:, tsteps_igr[4]] xs])
end

open(base_path * "ref_T1.csv", "w") do io
    writedlm(io, [u_ref[:, tsteps_ref[1]] ρ_ref[:, tsteps_ref[1]] xs_ref])
end

open(base_path * "ref_T2.csv", "w") do io
    writedlm(io, [u_ref[:, tsteps_ref[2]] ρ_ref[:, tsteps_ref[2]] xs_ref])
end

open(base_path * "ref_T3.csv", "w") do io
    writedlm(io, [u_ref[:, tsteps_ref[3]] ρ_ref[:, tsteps_ref[3]] xs_ref])
end

open(base_path * "ref_T4.csv", "w") do io
    writedlm(io, [u_ref[:, tsteps_ref[4]] ρ_ref[:, tsteps_ref[4]] xs_ref])
end