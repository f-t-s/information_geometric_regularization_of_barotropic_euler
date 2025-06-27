include("IGR1D.jl")
using DelimitedFiles
using StatsBase: mean

# problem size
L = 1.0
T = 0.055

# initial condition
β = 3.0
c = 0.0
s = 0.0
u0_funct, ρ0_funct = sine_wave(β, c, s, L)

# Determining the gas law
a = 1.0
γ = 1.4
ν = 0e-3


# Setting time step and running computation
Δt_record = 0.0001
# Whether to report the times at which snapshots are recorded
verbose = false
stype = Float16


# Reference solution: 
m_ref = 80000

xs_ref, Δx_ref, u0_ref, ρ0_ref = initialize(m_ref, L, u0_funct, ρ0_funct)
Δt_ref = Δx_ref / 4.25

sd_ref = SemidiscretizationRLW(a, γ, ν, Δx_ref, Δt_ref, m_ref)
@time u_ref, ρ_ref, ts_ref = forward_euler(sd_ref, Δt_ref, T, ρ0_ref, u0_ref, Δt_record, verbose)

u_errors = Real[]
ρ_errors = Real[]
μ_errors = Real[]
αs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

for α in αs
    sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α, Δx_ref, Δt_ref, m_ref)
    @time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_ref, T, ρ0_ref, u0_ref, Δt_record, verbose);

    push!(u_errors, mean(abs.(u_igr[:, end] - u_ref[:, end])) / mean(abs.(u_ref[:, end])))  
    push!(μ_errors, mean(abs.((u_igr .* ρ_igr)[:, end] - (u_ref .* ρ_ref)[:, end])) / mean(abs.((u_ref .* ρ_ref)[:, end])))  
    push!(ρ_errors, mean(abs.(ρ_igr[:, end] - ρ_ref[:, end])) / mean(abs.(ρ_ref[:, end])))  
end

base_path = "out/csv/convergence_study/"

open(base_path * "smooth_errors.csv", "w") do io
    writedlm(io, [αs u_errors μ_errors ρ_errors])
end

open(base_path * "smooth_ref_sol.csv", "w") do io
    writedlm(io, [xs_ref ρ_ref[:, end] u_ref[:, end]])
end