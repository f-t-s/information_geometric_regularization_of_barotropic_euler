include("IGR1D.jl")
using DelimitedFiles

L = 1.0 
m = 500
T = 0.5
Δt_record = 0.0001
verbose = false

β = 1.5
c = 0.0
s = 0.5
k = 1
u0_funct, ρ0_funct = riemann(β, c, s, 10)
xs, Δx, u0, ρ0, inds = initialize(m, L, u0_funct, ρ0_funct, 1)
m = length(xs)

# Determining the gas law
a = 0.2
γ = 1.4
ν = 0e-3
Δt_igr = Δx / 2.1
Δt_lad = Δx / 2.1

α_lad = 2.5 * Δx^2
α_igr= 2.5 * Δx^2
sd_igr = SemidiscretizationRLWIGR(a, γ, ν, α_igr, Δx, Δt_igr, m)
sd_lad = SemidiscretizationRLWLAD(a, γ, ν, α_lad, Δx, Δt_lad, m)

@time u_igr, ρ_igr, ts_igr = forward_euler(sd_igr, Δt_igr, T, ρ0, u0, Δt_record, verbose);

@time u_lad, ρ_lad, ts_lad = forward_euler(sd_lad, Δt_lad, T, ρ0, u0, Δt_record, verbose);

base_path = "out/csv/lad_comparison_oscillations/"

open(base_path * "igr.csv", "w") do io
    writedlm(io, [u_igr[inds, end] ρ_igr[inds, end] xs[inds]])
end

open(base_path * "lad.csv", "w") do io
    writedlm(io, [u_lad[inds, end] ρ_lad[inds, end] xs[inds]])
end
