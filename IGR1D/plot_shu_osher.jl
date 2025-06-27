include("IGR1D.jl")
using DelimitedFiles

# problem size
L = 1.0
m = 500
T = 2.0

# initial condition
L = 10
ρ_l = 2
ρ_r = 1
u_l = 3.0
u_r = 0
β = 0.2
k = 25
ϵ = 0.1
u0_funct, ρ0_funct = shu_osher(L, ρ_l, ρ_r, u_l, u_r, β, k, ϵ)
xs, Δx, u0, ρ0, inds = initialize(m, L, u0_funct, ρ0_funct, 1, 3.0)

m = length(xs)

# Determining the gas law
a = 1.0
γ = 1.4
ν = 0e-3
Δt_lf = Δx / 4.5
Δt_rlw = Δx / 4.5
Δt_igr = Δx / 4.5
Δt_lad = Δx / 4.5


α_igr = 20 * Δx ^ 2
α_lad = 4.5 * Δx ^ 2

# Setting time step and running computation
Δt_record = 0.0001
# Whether to report the times at which snapshots are recorded
verbose = false
stype = Float16

sd_rlw = SemidiscretizationRLW(a, γ, ν, Δx, Δt_rlw, m)
sd_lf = SemidiscretizationLF(a, γ, ν, Δx, Δt_lf, m)
sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α_igr, Δx, Δt_igr, m)
sd_lad = SemidiscretizationRLWLAD(a, γ, ν, α_lad, Δx, Δt_lad, m)

@time u_lf, ρ_lf, ts_lf = forward_euler(sd_lf, Δt_lf, T, ρ0, u0, Δt_record, verbose)
@time u_rlw, ρ_rlw, ts_rlw = forward_euler(sd_rlw, Δt_rlw, T, ρ0, u0, Δt_record, verbose);
@time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_igr, T, ρ0, u0, Δt_record, verbose);
@time u_lad, ρ_lad, ts_lad = forward_euler(sd_lad, Δt_lad, T, ρ0, u0, Δt_record, verbose);

# Reference solution: 
m_ref = 40000
xs_ref, Δx_ref, u0_ref, ρ0_ref, inds_ref = initialize(m_ref, L, u0_funct, ρ0_funct, 1, 3.0)
m_ref = length(xs_ref)
Δt_ref = Δx_ref / 4.5
sd_ref = SemidiscretizationLF(a, γ, ν, Δx_ref, Δt_ref, m_ref)
@time u_ref, ρ_ref, ts_ref = forward_euler(sd_ref, Δt_ref, T, ρ0_ref, u0_ref, Δt_record, verbose)


base_path = "out/csv/shu_osher/"

open(base_path * "lf.csv", "w") do io
    writedlm(io, [u_lf[inds, end] ρ_lf[inds, end] xs[inds]])
end

open(base_path * "rlw.csv", "w") do io
    writedlm(io, [u_rlw[inds, end] ρ_rlw[inds, end] xs[inds]])
end

open(base_path * "igr.csv", "w") do io
    writedlm(io, [u_igr[inds, end] ρ_igr[inds, end] xs[inds]])
end

open(base_path * "lad.csv", "w") do io
    writedlm(io, [u_lad[inds, end] ρ_lad[inds, end] xs[inds]])
end

open(base_path * "ref.csv", "w") do io
    writedlm(io, [u_ref[inds_ref, end] ρ_ref[inds_ref, end] xs_ref[inds_ref]])
end