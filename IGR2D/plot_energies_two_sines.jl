include("IGR2D.jl")
using Serialization
using DelimitedFiles
using CairoMakie
using LaTeXStrings
using FlorianStyle
set_theme!(FlorianStyle.empty_theme)
CairoMakie.activate!(px_per_unit=10.0)


Lx = 1.2
Ly = 1.0

base_out_path = "out/csv/two_sines_x/"

path_igr = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_600_n_500_T_0.4_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.00044444444444444447_integrator=rk2_method_IGRIter_alpha_5.0e-5_iters_1.jlsrl"

sk_igr = 1
ux_igr, uy_igr, ρ_igr, xs_igr, ys_igr, ts_igr, sd_igr = deserialize(path_igr)
ux_igr = ux_igr[1 : sk_igr : end, 1 : sk_igr : end, :]; uy_igr = uy_igr[1 : sk_igr : end, 1 : sk_igr : end, :]
ρ_igr = ρ_igr[1 : sk_igr : end, 1 : sk_igr : end, :]; 
xs_igr = xs_igr[1 : sk_igr : end]; ys_igr = ys_igr[1 : sk_igr : end]


path_lf = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_600_n_500_T_0.4_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.001_integrator=rk4_method_LF.jlsrl"

sk_lf = 1 
ux_lf, uy_lf, ρ_lf, xs_lf, ys_lf, ts_lf, sd_lf = deserialize(path_lf)
ux_lf = ux_lf[1 : sk_lf : end, 1 : sk_lf : end, :]; uy_lf = uy_lf[1 : sk_lf : end, 1 : sk_lf : end, :]
ρ_lf = ρ_lf[1 : sk_lf : end, 1 : sk_lf : end, :]; 
xs_lf = xs_lf[1 : sk_lf : end]; ys_lf = ys_lf[1 : sk_lf : end]

path_ref = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_12000_n_10000_T_0.4_dtrec_0.001_subsamp_5_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_3.9999999999999996e-5_integrator=rk4_method_LF.jlsrl"

ux_ref, uy_ref, ρ_ref, xs_ref, ys_ref, ts_ref, sd_ref = deserialize(path_ref)
sk_ref = 1
ux_ref = ux_ref[1 : sk_ref : end, 1 : sk_ref : end, :]; uy_ref = uy_ref[1 : sk_ref : end, 1 : sk_ref : end, :]
ρ_ref = ρ_ref[1 : sk_ref : end, 1 : sk_ref : end, :]; 
xs_ref = xs_ref[1 : sk_ref : end]; ys_ref = ys_ref[1 : sk_ref : end]

ekin_igr, epot_igr, equib_igr, etot_igr = compute_energies(sd_igr.sd1, ux_igr, uy_igr, ρ_igr)
ekin_lf, epot_lf, equib_lf, etot_lf = compute_energies(sd_lf, ux_lf, uy_lf, ρ_lf)
ekin_ref, epot_ref, equib_ref, etot_ref = compute_energies(sd_ref, ux_ref, uy_ref, ρ_ref)

open("out/csv/two_sine_waves/energy_lf_two_sine_waves.csv", "w") do io
    writedlm(io, [ekin_lf epot_lf .- equib_lf etot_lf .- equib_lf ts_lf])
end

open("out/csv/two_sine_waves/energy_igr_two_sine_waves.csv", "w") do io
    writedlm(io, [ekin_igr epot_igr .- equib_igr etot_igr .- equib_igr ts_igr])
end

open("out/csv/two_sine_waves/energy_ref_two_sine_waves.csv", "w") do io
    writedlm(io, [ekin_ref epot_ref .- equib_ref etot_ref .- equib_ref ts_ref])
end
