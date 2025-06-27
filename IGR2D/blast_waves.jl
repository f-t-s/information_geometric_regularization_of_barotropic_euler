include("IGR2D.jl")
using Serialization


# Setting the problem size
Lx = 0.72
Ly = 1.2
m = 2 * 216
n = 2 * 360
T = 0.4
Δt_record = 0.001
subsamp = 1

# Determining the gas law
a = 1.00
γ = 1.4
ν = 0e-4

# Computing the continuous initial conditions
βs  = [0.6, 1.2, 0.5]
cxs = [0.2, 0.4, 0.3]
cys = [0.2, 0.7, 1.05]
σs  = [0.05, 0.075, 0.03]
ρ_atmos = 1.0
ux0_funct, uy0_funct, ρ0_funct = sedov(βs, cxs, cys, σs, Lx, Ly, ρ_atmos)

# Creating the discrete initial conditions
xs, ys, Δx, Δy, ux0, uy0, ρ0 = initialize(m, n, Lx, Ly, ux0_funct, uy0_funct, ρ0_funct)
Δt = min(Δx, Δy) / 5.5

integrator = rk2
method = "IGRIter"
# settings for IGR
α = 1.0e-5
iters = 1

# Not part of the file path since it does not change the result
verbose = true

path  = "out/jlsrl/blast_waves/blast_waves_Lx_$(Lx)_Ly_$(Ly)_m_$(m)_n_$(n)_T_$(T)_dtrec_$(Δt_record)_subsamp_$(subsamp)_a_$(a)_gamma_$(γ)_nu_$(ν)_dt_$(Δt)_integrator=$(integrator)_method_$(method)"
if method == "IGRIter"
    # add the method-specific variables α and iters
    path = path * "_alpha_$(α)_iters_$(iters).jlsrl"
    sd_trlw = SemidiscretizationTRLW(a, γ, ν, Δx, Δy, Δt, m, n)
    # sd_igr_only = SemidiscretizationIGR(α, iters, Δx, Δy, Δt, m, n, ux0, uy0, ρ0)
    # sd_igr = DoubleSemidiscretization(sd_trlw, sd_igr_only, )
    sd_igr = SemidiscretizationTRLWIGR(a, γ, ν, α, iters, Δx, Δy, Δt, m, n, ux0, uy0, ρ0)
@time ux, uy, ρ, ts = integrator(sd_igr, Δt, T, ρ0, ux0, uy0, Δt_record, verbose, Float32, subsamp);
    xs = xs[1 : subsamp : end]
    ys = ys[1 : subsamp : end]
    serialize(path, (ux=ux, uy=uy, ρ=ρ, xs=xs, ys=ys, ts=ts, sd=sd_igr))
elseif method == "TRLW"
    sd_trlw = SemidiscretizationTRLW(a, γ, ν, Δx, Δy, Δt, m, n)
elseif method == "LF"
    path = path * ".jlsrl"
    sd_lf = SemidiscretizationLF(a, γ, ν, Δx, Δy, Δt, m, n)
    @time ux, uy, ρ, ts = integrator(sd_lf, Δt, T, ρ0, ux0, uy0, Δt_record, verbose, Float32, subsamp);
    xs = xs[1 : subsamp : end]
    ys = ys[1 : subsamp : end]
    serialize(path, (ux=ux, uy=uy, ρ=ρ, xs=xs, ys=ys, ts=ts, sd=sd_lf))
else 
    throw(error("Not a valid method!"))
end

plot_density(ρ, ts, xs, ys; ρlimits=(0.0, 18.0))
