include("IGR2D.jl")
using Serialization
using DelimitedFiles
using CairoMakie
using LaTeXStrings
using FlorianStyle
set_theme!(FlorianStyle.empty_theme)
CairoMakie.activate!(px_per_unit=10.0)
update_theme!(fontsize = 20)

Lx = 1.2
Ly = 1.0

base_out_path = "out/csv/two_sines_x/"

path_igr = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_600_n_500_T_3.0_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.00044444444444444447_integrator=rk2_method_IGRIter_alpha_5.0e-5_iters_1.jlsrl"

sk_igr = 1
ux_igr, uy_igr, ρ_igr, xs_igr, ys_igr, ts_igr, sd_igr = deserialize(path_igr)
ux_igr = ux_igr[1 : sk_igr : end, 1 : sk_igr : end, :]; uy_igr = uy_igr[1 : sk_igr : end, 1 : sk_igr : end, :]
ρ_igr = ρ_igr[1 : sk_igr : end, 1 : sk_igr : end, :]; 
xs_igr = xs_igr[1 : sk_igr : end]; ys_igr = ys_igr[1 : sk_igr : end]


path_lf = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_600_n_500_T_3.0_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.001_integrator=rk4_method_LF.jlsrl"

sk_lf = 1 
ux_lf, uy_lf, ρ_lf, xs_lf, ys_lf, ts_lf, sd_lf = deserialize(path_lf)
ux_lf = ux_lf[1 : sk_lf : end, 1 : sk_lf : end, :]; uy_lf = uy_lf[1 : sk_lf : end, 1 : sk_lf : end, :]
ρ_lf = ρ_lf[1 : sk_lf : end, 1 : sk_lf : end, :]; 
xs_lf = xs_lf[1 : sk_lf : end]; ys_lf = ys_lf[1 : sk_lf : end]


# creating plot using CairoMakie
fig_densities = Figure() 
 ax_lf_T1 = Axis(fig_densities[1, 1:2], ylabel=L"\text{LF + RK4}", title=L"\text{t = 1.0}")
ax_igr_T1 = Axis(fig_densities[2, 1:2], ylabel=L"\text{LW + IGR + RK2}")
                      
 ax_lf_T2 = Axis(fig_densities[1, 3:4], title=L"\text{t = 2.0}",)
ax_igr_T2 = Axis(fig_densities[2, 3:4])
                      
 ax_lf_T3 = Axis(fig_densities[1, 5:6], title=L"\text{t = 3.0}")
ax_igr_T3 = Axis(fig_densities[2, 5:6])

# colsize!(fig_densities.layout, 1, Aspect(1, 0.6))
# colsize!(fig_densities.layout, 2, Aspect(1, 0.6))
# colsize!(fig_densities.layout, 3, Aspect(1, 0.6))
# colsize!(fig_densities.layout, 4, Aspect(1, 0.6))
# colsize!(fig_densities.layout, 5, Aspect(1, 0.6))
# colsize!(fig_densities.layout, 6, Aspect(1, 0.6))

rowsize!(fig_densities.layout, 1, Aspect(1, 1 / 0.6))
rowsize!(fig_densities.layout, 2, Aspect(1, 1 / 0.6))

Ts = [1.0, 2.0, 3.0]
tsteps_igr = [findfirst(k -> ts_igr[k] ≥ T, eachindex(ts_igr)) for T in Ts]
tsteps_lf = [findfirst(k -> ts_lf[k] ≥ T, eachindex(ts_lf)) for T in Ts]

crange = (0.0, 2.0)
cmap=cgrad([FlorianStyle.seagreen, FlorianStyle.steelblue , :white], [0.0, 1.0, 2.0])
rstr = 5
heatmap!(ax_lf_T1, xs_lf, ys_lf, ρ_lf[:, :, tsteps_lf[1]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_lf_T2, xs_lf, ys_lf, ρ_lf[:, :, tsteps_lf[2]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_lf_T3, xs_lf, ys_lf, ρ_lf[:, :, tsteps_lf[3]], colorrange=crange, colormap=cmap, rasterize=rstr)

heatmap!(ax_igr_T1, xs_igr, ys_igr, ρ_igr[:, :, tsteps_igr[1]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_igr_T2, xs_igr, ys_igr, ρ_igr[:, :, tsteps_igr[2]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_igr_T3, xs_igr, ys_igr, ρ_igr[:, :, tsteps_igr[3]], colorrange=crange, colormap=cmap, rasterize=rstr)

cbar = Colorbar(fig_densities[3, 2:5], limits=crange, colormap=cmap, vertical=false, ticks=([0.5, 1.0, 1.5], [L"0.5", L"1.0", L"1.5"]))
resize_to_layout!(fig_densities)

colgap!(fig_densities.layout, 5)
rowgap!(fig_densities.layout, 5)
rowgap!(fig_densities.layout, 2, 10)
resize_to_layout!(fig_densities)

save("out/pdf/sine_waves_longtime_densities.pdf", fig_densities)
fig_densities
