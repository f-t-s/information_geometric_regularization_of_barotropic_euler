include("IGR2D.jl")
using Serialization


# Setting the problem size
Lx = 1.2
Ly = 1.0
m = 8 * 300
n = 8 * 250
T = 0.4
Δt_record = 0.001
subsamp = 1

# Determining the gas law
a = 1.00
γ = 1.4
ν = 0e-4

# Computing the continuous initial conditions
β1 = 2.5
c1 = 1.00
# s1 = 0.1
s1 = 1.0
β2 = -1.5
c2 = -1.00
# s2 = - 0.2
s2 = 0.9
ϵ = 0.05
ux0_funct, uy0_funct, ρ0_funct = two_sines_x(β1, c1, s1, β2, c2, s2, Lx, Ly, ϵ, 0.0)

# Creating the discrete initial conditions
xs, ys, Δx, Δy, ux0, uy0, ρ0 = initialize(m, n, Lx, Ly, ux0_funct, uy0_funct, ρ0_funct)
Δt = min(Δx, Δy) / 4.5

integrator = rk2
method = "IGRIter"
# settings for IGR
α = 3.125e-6 
iters = 1

# Not part of the file path since it does not change the result
verbose = true

path  = "out/jlsrl/two_sines_x/two_sines_x_Lx_$(Lx)_Ly_$(Ly)_m_$(m)_n_$(n)_T_$(T)_dtrec_$(Δt_record)_subsamp_$(subsamp)_a_$(a)_gamma_$(γ)_nu_$(ν)_beta1_$(β1)_c1_$(c1)_s1_$(s1)_beta2_$(β2)_c2_$(c2)_s2_$(s2)_eps_$(ϵ)_dt_$(Δt)_integrator=$(integrator)_method_$(method)"
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


plot_density(ρ, ts, xs, ys; ρlimits=(0.0, 3.0))