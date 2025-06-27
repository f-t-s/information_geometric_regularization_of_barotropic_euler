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

path_low = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_300_n_250_T_0.4_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.0008888888888888889_integrator=rk2_method_IGRIter_alpha_5.0e-5_iters_1.jlsrl"

path_med = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_600_n_500_T_0.4_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.00044444444444444447_integrator=rk2_method_IGRIter_alpha_5.0e-5_iters_1.jlsrl"

path_high = "out/jlsrl/two_sines_x/two_sines_x_Lx_1.2_Ly_1.0_m_1200_n_1000_T_0.4_dtrec_0.001_subsamp_1_a_1.0_gamma_1.4_nu_0.0_beta1_2.5_c1_1.0_s1_1.0_beta2_-1.5_c2_-1.0_s2_0.9_eps_0.05_dt_0.00022222222222222223_integrator=rk2_method_IGRIter_alpha_5.0e-5_iters_1.jlsrl"


fig = Figure() 
xmin = 0.15; xmax = 0.75
ymin = 0.35; ymax = 0.85
ax_low = Axis(fig[1, 1], title = L"\Delta x = \Delta y = 0.004", ylabel=L"\text{LW + IGR + RK2}")
xlims!(xmin, xmax)
ylims!(ymin, ymax)
ax_med = Axis(fig[1, 2], title = L"\Delta x = \Delta y = 0.002")
xlims!(xmin, xmax)
ylims!(ymin, ymax)
ax_high= Axis(fig[1, 3], title = L"\Delta x = \Delta y = 0.001")
xlims!(xmin, xmax)
ylims!(ymin, ymax)

rowsize!(fig.layout, 1, Aspect(1, 1 / 1.2))
colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)
resize_to_layout!(fig)


Ts = [0.25]
ux_low, uy_low, ρ_low, xs_low, ys_low, ts_low, sd_low = deserialize(path_low)
ux_med, uy_med, ρ_med, xs_med, ys_med, ts_med, sd_med = deserialize(path_med)
ux_high, uy_high, ρ_high, xs_high, ys_high, ts_high, sd_high = deserialize(path_high)
tsteps_low = [findfirst(k -> ts_low[k] ≥ T, eachindex(ts_low)) for T in Ts]
tsteps_med = [findfirst(k -> ts_med[k] ≥ T, eachindex(ts_med)) for T in Ts]
tsteps_high= [findfirst(k -> ts_high[k] ≥ T, eachindex(ts_high)) for T in Ts]

crange = (0.0, 5.0)
cmap=cgrad([FlorianStyle.darksky, FlorianStyle.steelblue , :white], [0.0, 2.5, 5.0])
rstr = 5
heatmap!(ax_low, xs_low, ys_low, ρ_low[:, :, tsteps_low[1]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_med, xs_med, ys_med, ρ_med[:, :, tsteps_med[1]], colorrange=crange, colormap=cmap, rasterize=rstr)
heatmap!(ax_high, xs_high, ys_high, ρ_high[:, :, tsteps_high[1]], colorrange=crange, colormap=cmap, rasterize=rstr)

# cbar = Colorbar(fig[2, 2:5], limits=crange, colormap=cmap, vertical=false, ticks=([1, 4], [L"\text{low density}", L"\text{high density}"]), ticksize=0)

save("out/pdf/sine_waves_mesh_convergence.pdf", fig)