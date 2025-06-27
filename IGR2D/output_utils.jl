"""
    Writes a density to csv in a form accepted by pgfplots surf.
    assumes that the y coordinate is the slow-changing one.
    Modified from https://discourse.julialang.org/t/writedlm-adding-a-blank-line-every-n-lines/36111
"""
function write_density_to_pgf(path, xs, ys, ρ)
    open(path, "w") do io
        m = length(xs)
        n = length(ys)
        x_coords_igr = vec(repeat(xs', n))
        y_coords_igr = repeat(ys, m) 
        ρ = vec(ρ')
    	last=y_coords_igr[firstindex(x_coords_igr)]
    	for index in eachindex(x_coords_igr)
    		if last != x_coords_igr[index]
    			writedlm(io," ")
    		end
    		writedlm(io, [x_coords_igr[index] y_coords_igr[index] ρ[index]])
    		last=x_coords_igr[index]
    	end
    end
end



# computes the potential, kinetic, and total energies
function compute_energies(sd, ux, uy, ρ)
    nx = size(ρ, 1)
    ny = size(ρ, 2)
    nt = size(ρ, 3)
    kinetic_energies = [sum((ux[:, :, i].^2 .+ uy[:, :, i].^2) .* ρ[:, :, i]) / 2 for i = 1 : nt] / nx / ny
    # Computing the potential energy at each time step
    potential_energies = [sum(e.(ρ[:, :, i], [sd]) .* ρ[:, :, i]) for i = 1 : nt] / nx / ny
    equilibrium_energy = sum(ρ[:, :, 1]) * e(sum(ρ[:, :, 1]) / length(ρ[:, :, 1]), sd) / nx / ny
    total_energies = kinetic_energies + potential_energies
    return kinetic_energies, potential_energies, equilibrium_energy, total_energies
end
