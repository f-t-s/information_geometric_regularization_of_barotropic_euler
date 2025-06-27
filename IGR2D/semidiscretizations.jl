"""
    A semidiscretization implements the f in 
    ẋ(t) = f(x) and serves as the input for time steppers
"""
abstract type AbstractSemidiscretization end

"""
    A semidiscretization implements the f in 
    ẋ(t) = f(x) and serves as the input for time steppers
"""
abstract type AbstractBarotropicSemidiscretization<:AbstractSemidiscretization end

struct SemidiscretizationTRLW<:AbstractBarotropicSemidiscretization
    # We compute the pressure as P(ρ) = a ρ^γ
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float

    # The spatial discretization sizes
    Δx::Float
    # The spatial discretization sizes
    Δy::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Wendroff method 
    fxμx::Matrix{Float}
    fyμx::Matrix{Float}
    fxμy::Matrix{Float}
    fyμy::Matrix{Float}
    fxρ ::Matrix{Float}
    fyρ ::Matrix{Float}

    # Storage for the intermediate step of TRLW 
    μx_inter::Matrix{Float}
    μy_inter::Matrix{Float}
    ρ_inter ::Matrix{Float}
end

"""
    Constructor for TRLW semidiscretization 
"""
function SemidiscretizationTRLW(a, γ, ν, Δx, Δy, Δt, m, n)
    return SemidiscretizationTRLW(a, γ, ν, Δx, Δy, Δt,
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n))
end

"""
    Extracts coefficients for gas law 
"""
function get_gas_law(sd::AbstractBarotropicSemidiscretization)
    return (a=sd.a, γ=sd.γ, ν=sd.ν)
end

"""
    This function computes the pressure in a given point
"""
function p(ρ, sd::AbstractBarotropicSemidiscretization)
    @assert ρ ≥ 0
    law = get_gas_law(sd)
    return law.a * ρ^law.γ
end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_size(sd::AbstractBarotropicSemidiscretization)
    return size(sd.fxμx)
end

"""
    This function computes the internal energy in a given point
"""
function e(ρ, sd::AbstractBarotropicSemidiscretization)
    @assert ρ ≥ 0
    law = get_gas_law(sd)
    return law.a * ρ^law.γ / (law.γ - 1) / ρ
end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_Δs(sd::AbstractSemidiscretization)
    return sd.Δx, sd.Δy, sd.Δt
end

"""
    This function computes the fluxes (of sd) in place.
"""
function compute_fluxes!(sd::AbstractBarotropicSemidiscretization, μx, μy, ρ)
    # Ensures that all fluxes have the correct size
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  

    # The viscosity parameter
    ν = get_gas_law(sd)[3]
    
    @fastmath @inbounds Threads.@threads for i in eachindex(fxμy)
        pressure = p(ρ[i], sd)
        # Computing the momentum fluxes. 
        fxμx[i] = μx[i] * μx[i] / ρ[i] + pressure;        fxμy[i] = μx[i] .* μy[i] ./ ρ[i]
        fyμx[i] = μy[i] * μx[i] / ρ[i];                   fyμy[i] = μy[i] .* μy[i] ./ ρ[i] + pressure 
        # Computing the mass fluxes.
        fxρ[i]  = μx[i];                                  fyρ[i]  = μy[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m, n = get_size(sd)
        Δx, Δy = get_Δs(sd)
        for i in 1 : m, j = 1 : n
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1
            
            ∂xux = (μx[iu, j] / ρ[iu, j] - μx[id, j] / ρ[id, j]) / 2 / Δx
            ∂xuy = (μy[iu, j] / ρ[iu, j] - μy[id, j] / ρ[id, j]) / 2 / Δx

            ∂yux = (μx[i, ju] / ρ[i, ju] - μx[i, jd] / ρ[i, jd]) / 2 / Δy
            ∂yuy = (μy[i, ju] / ρ[i, ju] - μy[i, jd] / ρ[i, jd]) / 2 / Δy

            fxμx[i, j] -= ν * (∂xux + ∂xux) / 2
            fyμy[i, j] -= ν * (∂yuy + ∂yuy) / 2
            fxμy[i, j] -= ν * (∂xuy + ∂yux) / 2
            fyμx[i, j] -= ν * (∂yux + ∂xuy) / 2
        end
    end
end

"""
    This function computes only the x fluxes (of sd) in place.
"""
function compute_x_fluxes!(sd::AbstractBarotropicSemidiscretization, μx, μy, ρ)
    # Ensures that all fluxes have the correct size
    fxμx = sd.fxμx 
    fxμy = sd.fxμy 
    fxρ  = sd.fxρ  

    # The viscosity parameter
    ν = get_gas_law(sd)[3]

    Threads.@threads for i in eachindex(fxμy)
        pressure = p(ρ[i], sd)
        # Computing the momentum fluxes. 
        fxμx[i] = μx[i] * μx[i] / ρ[i] + pressure;
        fxμy[i] = μx[i] .* μy[i] ./ ρ[i]
        # Computing the mass fluxes.
        fxρ[i]  = μx[i];
    end

    # Computing the viscosity
    if ν ≠ 0 
        m, n = get_size(sd)
        Δx, Δy = get_Δs(sd)
        for i in 1 : m, j = 1 : n
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1
            
            ∂xux = (μx[iu, j] / ρ[iu, j] - μx[id, j] / ρ[id, j]) / 2 / Δx
            ∂xuy = (μy[iu, j] / ρ[iu, j] - μy[id, j] / ρ[id, j]) / 2 / Δx

            ∂yux = (μx[i, ju] / ρ[i, ju] - μx[i, jd] / ρ[i, jd]) / 2 / Δy

            fxμx[i, j] -= ν * (∂xux + ∂xux) / 2
            fxμy[i, j] -= ν * (∂xuy + ∂yux) / 2
        end
    end
end

"""
    This function computes only the y fluxes (of sd) in place.
"""
function compute_y_fluxes!(sd::AbstractBarotropicSemidiscretization, μx, μy, ρ)
    # Ensures that all fluxes have the correct size
    fyμx = sd.fyμx 
    fyμy = sd.fyμy 
    fyρ  = sd.fyρ  

    # The viscosity parameter
    ν = get_gas_law(sd)[3]

    Threads.@threads for i in eachindex(fyμy)
        pressure = p(ρ[i], sd)
        # Computing the momentum fluxes. 
        fyμx[i] = μy[i] * μx[i] / ρ[i];
        fyμy[i] = μy[i] .* μy[i] ./ ρ[i] + pressure 
        # Computing the mass fluxes.
        fyρ[i]  = μy[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m, n = get_size(sd)
        Δx, Δy = get_Δs(sd)
        for i in 1 : m, j = 1 : n
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1
            
            ∂xuy = (μy[iu, j] / ρ[iu, j] - μy[id, j] / ρ[id, j]) / 2 / Δx

            ∂yux = (μx[i, ju] / ρ[i, ju] - μx[i, jd] / ρ[i, jd]) / 2 / Δy
            ∂yuy = (μy[i, ju] / ρ[i, ju] - μy[i, jd] / ρ[i, jd]) / 2 / Δy

            fyμy[i, j] -= ν * (∂yuy + ∂yuy) / 2
            fyμx[i, j] -= ν * (∂yux + ∂xuy) / 2
        end
    end
end



"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::SemidiscretizationTRLW,
                  μx_old, μy_old, ρ_old)
    m, n = get_size(sd)
    Δx, Δy, Δt = get_Δs(sd)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  
    μx_inter = sd.μx_inter
    μy_inter = sd.μy_inter
    ρ_inter = sd.ρ_inter

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μx_old, μy_old, ρ_old)

    Threads.@threads for j = 1 : n 
        @fastmath @inbounds @simd for i = 1 : m
            # Compute the indices before and after, accounting for periodicity
            # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
            # Accordingly, id = i, iu = u etc, since they correspond to half-step
            # decrements of from the intermediate index i positioned in x[i + 1/2] 
            id = i
            iu = mod(i, m) + 1
            jd = j
            ju = mod(j, n) + 1

            μx_inter[i, j] = ((μx_old[iu, ju] + μx_old[iu, jd] + μx_old[id, ju] + μx_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxμx[iu, jd] + fxμx[iu, ju] - fxμx[id, jd] - fxμx[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyμx[id, ju] + fyμx[iu, ju] - fyμx[id, jd] - fyμx[iu, jd]))

            μy_inter[i, j] = ((μy_old[iu, ju] + μy_old[iu, jd] + μy_old[id, ju] + μy_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxμy[iu, jd] + fxμy[iu, ju] - fxμy[id, jd] - fxμy[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyμy[id, ju] + fyμy[iu, ju] - fyμy[id, jd] - fyμy[iu, jd]))

            ρ_inter[i, j]  = ((ρ_old[iu, ju] + ρ_old[iu, jd] + ρ_old[id, ju] + ρ_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxρ[iu, jd] + fxρ[iu, ju] - fxρ[id, jd] - fxρ[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyρ[id, ju] + fyρ[iu, ju] - fyρ[id, jd] - fyρ[iu, jd]))
        end
    end

    # Compute the fluxes at the predictor point
    compute_fluxes!(sd, μx_inter, μy_inter, ρ_inter)

    Threads.@threads for j = 1 : n
        @fastmath @inbounds @simd  for i = 1 : m
            # Compute the indices before and after, accounting for periodicity
            # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
            # The indices for the fluxes around a location [i, j] in the primal mesh are
            # thus given as below. In the below, we are now re-using the *_new variables
            # as primal points 
            id = mod(i - 2, m) + 1
            iu = i
            jd = mod(j - 2, n) + 1
            ju = j

            # We are not multiplying with Δt, since this is done by the time stepper
            μx_out[i, j] = (  μx_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμx[iu, jd] + fxμx[iu, ju] - fxμx[id, jd] - fxμx[id, ju]) 
                            - (1 / (2 * Δy)) * (fyμx[id, ju] + fyμx[iu, ju] - fyμx[id, jd] - fyμx[iu, jd]))

            μy_out[i, j] = (  μy_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμy[iu, jd] + fxμy[iu, ju] - fxμy[id, jd] - fxμy[id, ju]) 
                            - (1 / (2 * Δy)) * (fyμy[id, ju] + fyμy[iu, ju] - fyμy[id, jd] - fyμy[iu, jd]))

            ρ_out[i, j]  = (  ρ_out[i, j] 
                            - (1 / (2 * Δx)) * (fxρ[iu, jd] + fxρ[iu, ju] - fxρ[id, jd] - fxρ[id, ju]) 
                            - (1 / (2 * Δy)) * (fyρ[id, ju] + fyρ[iu, ju] - fyρ[id, jd] - fyρ[iu, jd]))
       end
    end
end

struct SemidiscretizationIGR{IGRFluxType<:AbstractIGRFlux}<:AbstractSemidiscretization
    # The weight of the regularization 
    α::Float

    # The spatial discretization sizes
    Δx::Float
    # The spatial discretization sizes
    Δy::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # currently reusing the IGRflux definition
    flux::IGRFluxType

    # Storage for the fluxes computed during the Lax Wendroff method 
    fxμx::Matrix{Float}
    fyμx::Matrix{Float}
    fxμy::Matrix{Float}
    fyμy::Matrix{Float}
    fxρ ::Matrix{Float}
    fyρ ::Matrix{Float}

    # Storage for the intermediate step of TRLW 
    μx_inter::Matrix{Float}
    μy_inter::Matrix{Float}
    ρ_inter ::Matrix{Float}
end


"""
    Constructor for IGR semidiscretization with IGRFluxLU
"""
function SemidiscretizationIGR(α, Δx, Δy, Δt, m, n)
    flux = IGRFluxLU(α, Δx, Δy, m, n)

    return SemidiscretizationIGR{IGRFluxLU}(α, Δx, Δy, Δt, flux, 
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n))
    
end

"""
    Constructor for IGR semidiscretization with IGRFluxLU
"""
function SemidiscretizationIGR(α, max_iter, Δx, Δy, Δt, m, n, ux0, uy0, ρ0, first_max_iter = 100 * max_iter)


    # just because the constructor of IGR flux still requires one (to be changed)
    # flux = IGRFluxGS(α, max_iter, Δx, Δy, m, n)
    flux = IGRFluxJI(α, max_iter, Δx, Δy, m, n)

    # Uses LU flux in order to initialize Σ
    aux_flux = IGRFluxJI(α, first_max_iter, Δx, Δy, m, n)
    compute_fluxes!(zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), ux0 .* ρ0, uy0 .* ρ0, ρ0, aux_flux)
    flux.Σ .= aux_flux.Σ


    # return SemidiscretizationIGR{IGRFluxGS}(α, Δx, Δy, Δt, flux, 
    return SemidiscretizationIGR{IGRFluxJI}(α, Δx, Δy, Δt, flux, 
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n))
end


"""
    This function returns the size of the grid of a semidiscretization
"""
# function get_size(sd::SemidiscretizationIGR{<:AbstractIGRFlux})
function get_size(sd::SemidiscretizationIGR{<:AbstractIGRFlux})
    return size(sd.fxμx)
end


"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::SemidiscretizationIGR{<:AbstractIGRFlux},
                  μx_old, μy_old, ρ_old)
    m, n = get_size(sd)
    Δx, Δy, Δt = get_Δs(sd)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  

    # compute the IGR flux
    compute_fluxes!(fxμx, fyμx, fxμy, fyμy, fxρ, fyρ, μx_old, μy_old, ρ_old, sd.flux)

    # using fluxes via central differences to update the state
    Threads.@threads for j = 1 : n 
        @fastmath @inbounds @simd for i = 1 : m 
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1

            μx_out[i, j] = (  μx_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμx[iu, j] - fxμx[id, j]) 
                            - (1 / (2 * Δy)) * (fyμx[i, ju] - fyμx[i, jd]))

            μy_out[i, j] = (  μy_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμy[iu, j] - fxμy[id, j]) 
                            - (1 / (2 * Δy)) * (fyμy[i, ju] - fyμy[i, jd]))

            ρ_out[i, j]  = (  ρ_out[i, j] 
                            - (1 / (2 * Δx)) * (fxρ[iu, j] - fxρ[id, j]) 
                            - (1 / (2 * Δy)) * (fyρ[i, ju] - fyρ[i, jd]))
        end
    end
end


struct DoubleSemidiscretization
    sd1
    sd2
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::DoubleSemidiscretization,
                  μx_old, μy_old, ρ_old)
    add_rhs!(μx_out, μy_out, ρ_out, sd.sd1,
             μx_old, μy_old, ρ_old)
    add_rhs!(μx_out, μy_out, ρ_out, sd.sd2,
             μx_old, μy_old, ρ_old)
end

"""
    The Lax-Wendroff scheme of Zwas 1991 
"""
struct SemidiscretizationZRLW<:AbstractBarotropicSemidiscretization
    # We compute the pressure as P(ρ) = a ρ^γ
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float

    # The spatial discretization sizes
    Δx::Float
    # The spatial discretization sizes
    Δy::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Wendroff method 
    fxμx::Matrix{Float}
    fyμx::Matrix{Float}
    fxμy::Matrix{Float}
    fyμy::Matrix{Float}
    fxρ ::Matrix{Float}
    fyρ ::Matrix{Float}

    # Storage for the x and y directional means
    μx_mean_x::Matrix{Float}
    μy_mean_x::Matrix{Float}
    ρ_mean_x ::Matrix{Float}

    μx_mean_y::Matrix{Float}
    μy_mean_y::Matrix{Float}
    ρ_mean_y ::Matrix{Float}


    # Storage for the intermediate step of TRLW 
    μx_inter::Matrix{Float}
    μy_inter::Matrix{Float}
    ρ_inter ::Matrix{Float}
end

"""
    Constructor for ZRLW semidiscretization 
"""
function SemidiscretizationZRLW(a, γ, ν, Δx, Δy, Δt, m, n)
    return SemidiscretizationZRLW(a, γ, ν, Δx, Δy, Δt,
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n))
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::SemidiscretizationZRLW,
                  μx_old, μy_old, ρ_old)
    m, n = get_size(sd)
    Δx, Δy, Δt = get_Δs(sd)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  
    μx_mean_x = sd.μx_mean_x
    μy_mean_x = sd.μy_mean_x
     ρ_mean_x = sd.ρ_mean_x
    μx_mean_y = sd.μx_mean_y
    μy_mean_y = sd.μy_mean_y
     ρ_mean_y = sd.ρ_mean_y
    μx_inter = sd.μx_inter
    μy_inter = sd.μy_inter
     ρ_inter = sd.ρ_inter

    # Compute the x and y directional means
    for i = 1 : m, j = 1 : n
        id = i
        iu = mod(i, m) + 1
        jd = j
        ju = mod(j, n) + 1

        μx_mean_x[i, j] = (μx_old[i, j] + μx_old[iu, jd]) / 2
        μy_mean_x[i, j] = (μy_old[i, j] + μy_old[iu, jd]) / 2
         ρ_mean_x[i, j] = ( ρ_old[i, j] +  ρ_old[iu, jd]) / 2

        μx_mean_y[i, j] = (μx_old[i, j] + μx_old[id, ju]) / 2
        μy_mean_y[i, j] = (μy_old[i, j] + μy_old[id, ju]) / 2
         ρ_mean_y[i, j] = ( ρ_old[i, j] +  ρ_old[id, ju]) / 2
    end
    # Compute the intermediate fluxes 
    compute_x_fluxes!(sd, μx_mean_y, μy_mean_y, ρ_mean_y)
    compute_y_fluxes!(sd, μx_mean_x, μy_mean_x, ρ_mean_x)

    for i = 1 : m, j = 1 : n
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2]
        id = i
        iu = mod(i, m) + 1
        jd = j
        ju = mod(j, n) + 1

        μx_inter[i, j] = ((μx_old[iu, ju] + μx_old[iu, jd] + μx_old[id, ju] + μx_old[id, jd]) / 4 
                        - (Δt / (2 * Δx)) * (fxμx[iu, j] - fxμx[id, j]) 
                        - (Δt / (2 * Δy)) * (fyμx[i, ju] - fyμx[i, jd]))

        μy_inter[i, j] = ((μy_old[iu, ju] + μy_old[iu, jd] + μy_old[id, ju] + μy_old[id, jd]) / 4 
                        - (Δt / (2 * Δx)) * (fxμy[iu, j] - fxμy[id, j]) 
                        - (Δt / (2 * Δy)) * (fyμy[i, ju] - fyμy[i, jd]))

        ρ_inter[i, j]  = ((ρ_old[iu, ju] + ρ_old[iu, jd] + ρ_old[id, ju] + ρ_old[id, jd]) / 4 
                        - (Δt / (2 * Δx)) * (fxρ[iu, j] - fxρ[id, j]) 
                        - (Δt / (2 * Δy)) * (fyρ[i, ju] - fyρ[i, jd]))
    end

    # Compute the x and y directional means
    for i = 1 : m, j = 1 : n
        id = mod(i - 2, m) + 1
        iu = i
        jd = mod(j - 2, n) + 1
        ju = j

        μx_mean_x[i, j] = (μx_inter[i, j] + μx_inter[id, j ]) / 2
        μy_mean_x[i, j] = (μy_inter[i, j] + μy_inter[id, j ]) / 2
         ρ_mean_x[i, j] = ( ρ_inter[i, j] +  ρ_inter[id, j ]) / 2

        μx_mean_y[i, j] = (μx_inter[i, j] + μx_inter[i , jd]) / 2
        μy_mean_y[i, j] = (μy_inter[i, j] + μy_inter[i , jd]) / 2
         ρ_mean_y[i, j] = ( ρ_inter[i, j] +  ρ_inter[i , jd]) / 2
    end
    # Compute the intermediate fluxes 
    compute_x_fluxes!(sd, μx_mean_y, μy_mean_y, ρ_mean_y)
    compute_y_fluxes!(sd, μx_mean_x, μy_mean_x, ρ_mean_x)

    for i = 1 : m, j = 1 : n
        # Compute the indices before and after, accounting for periodicity
        # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
        # The indices for the fluxes around a location [i, j] in the primal mesh are
        # thus given as below. In the below, we are now re-using the *_new variables
        # as primal points 
        id = mod(i - 2, m) + 1
        iu = i
        jd = mod(j - 2, n) + 1
        ju = j

        # We are not multiplying with Δt, since this is done by the time stepper
        μx_out[i, j] = (μx_out[i, j] 
                        - (1 / (Δx)) * (fxμx[iu, j] - fxμx[id, j]) 
                        - (1 / (Δy)) * (fyμx[i, ju] - fyμx[i, jd]))

        μy_out[i, j] = (μy_out[i, j]
                        - (1 / (Δx)) * (fxμy[iu, j] - fxμy[id, j]) 
                        - (1 / (Δy)) * (fyμy[i, ju] - fyμy[i, jd]))

        ρ_out[i, j]  = (ρ_out[i, j] 
                        - (1 / (Δx)) * (fxρ[iu, j] - fxρ[id, j]) 
                        - (1 / (Δy)) * (fyρ[i, ju] - fyρ[i, jd]))
    end
end

"""
    Semidiscretization for the Lax-Friedrichs method
"""
struct SemidiscretizationLF<:AbstractBarotropicSemidiscretization
    # We compute the pressure as P(ρ) = a ρ^γ
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float

    # The spatial discretization sizes
    Δx::Float
    # The spatial discretization sizes
    Δy::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Friedrichs method 
    fxμx::Matrix{Float}
    fyμx::Matrix{Float}
    fxμy::Matrix{Float}
    fyμy::Matrix{Float}
    fxρ ::Matrix{Float}
    fyρ ::Matrix{Float}
end

"""
    Constructor for LF semidiscretization that creates the scratch
    spaces automatically 
"""
function SemidiscretizationLF(a, γ, ν, Δx, Δy, Δt, m, n)
    return SemidiscretizationLF(a, γ, ν, Δx, Δy, Δt,
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n),
                                  zeros(Float, m, n))
end


"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::SemidiscretizationLF,
                  μx_old, μy_old, ρ_old)
    m, n = get_size(sd)
    Δx, Δy, Δt = get_Δs(sd)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μx_old, μy_old, ρ_old)

    Threads.@threads for j = 1 : n 
        @fastmath @inbounds @simd for i = 1 : m
            # Compute the indices before and after, accounting for periodicity
            # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
            # Accordingly, id = i, iu = u etc, since they correspond to half-step
            # decrements of from the intermediate index i positioned in x[i + 1/2] 
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1

            μx_out[i, j] += (((μx_old[iu, ju] + μx_old[iu, jd] + μx_old[id, ju] + μx_old[id, jd]) / 4 - μx_old[i, j]) / Δt
                            - (1 / (2 * Δx)) * (fxμx[iu, j] - fxμx[id, j])
                            - (1 / (2 * Δy)) * (fyμx[i, ju] - fyμx[i, jd]))
            μy_out[i, j] += (((μy_old[iu, ju] + μy_old[iu, jd] + μy_old[id, ju] + μy_old[id, jd]) / 4 - μy_old[i, j]) / Δt
                            - (1 / (2 * Δx)) * (fxμy[iu, j] - fxμy[id, j])
                            - (1 / (2 * Δy)) * (fyμy[i, ju] - fyμy[i, jd]))
            ρ_out[i, j]  += (((ρ_old[iu, ju] + ρ_old[iu, jd] + ρ_old[id, ju] + ρ_old[id, jd]) / 4 - ρ_old[i, j]) / Δt
                            - (1 / (2 * Δx)) * (fxρ[iu, j] - fxρ[id, j])
                            - (1 / (2 * Δy)) * (fyρ[i, ju] - fyρ[i, jd]))
        end
    end
end

struct SemidiscretizationTRLWIGR{IGRFluxType<:AbstractIGRFlux}<:AbstractBarotropicSemidiscretization
    # The gas parameters    
    a::Float

    γ::Float

    ν::Float

    # The weight of the regularization 
    α::Float

    # The spatial discretization sizes
    Δx::Float
    # The spatial discretization sizes
    Δy::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # currently reusing the IGRflux definition
    flux::IGRFluxType

    # Storage for the fluxes computed during the Lax Wendroff method 
    fxμx::Matrix{Float}
    fyμx::Matrix{Float}
    fxμy::Matrix{Float}
    fyμy::Matrix{Float}
    fxρ ::Matrix{Float}
    fyρ ::Matrix{Float}

    # Storage for the intermediate step of TRLW 
    μx_inter::Matrix{Float}
    μy_inter::Matrix{Float}
    ρ_inter ::Matrix{Float}
end


"""
    Constructor for IGR semidiscretization with IGRFluxLU
"""
function SemidiscretizationTRLWIGR(a, γ, ν, α, Δx, Δy, Δt, m, n)
    flux = IGRFluxLU(α, Δx, Δy, m, n)

    return SemidiscretizationTRLWIGR{IGRFluxLU}(a, γ, ν, α, Δx, Δy, Δt, flux, 
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n))
    
end

"""
    Constructor for IGR semidiscretization with IGRFluxLU
"""
function SemidiscretizationTRLWIGR(a, γ, ν, α, max_iter, Δx, Δy, Δt, m, n, ux0, uy0, ρ0, first_max_iter = 100 * max_iter)


    # just because the constructor of IGR flux still requires one (to be changed)
    # flux = IGRFluxGS(α, max_iter, Δx, Δy, m, n)
    flux = IGRFluxJI(α, max_iter, Δx, Δy, m, n)

    # Uses LU flux in order to initialize Σ
    aux_flux = IGRFluxJI(α, first_max_iter, Δx, Δy, m, n)
    compute_fluxes!(zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), ux0 .* ρ0, uy0 .* ρ0, ρ0, aux_flux)
    flux.Σ .= aux_flux.Σ


    # return SemidiscretizationIGR{IGRFluxGS}(α, Δx, Δy, Δt, flux, 
    return SemidiscretizationTRLWIGR{IGRFluxJI}(a, γ, ν, α, Δx, Δy, Δt, flux, 
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n),
                                                        zeros(Float, m, n))
end

function compute_fluxes!(sd::SemidiscretizationTRLWIGR, μx, μy, ρ)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  

    # compute the fluxes due to IGR
    compute_fluxes!(fxμx, fyμx, fxμy, fyμy, fxρ, fyρ, μx, μy, ρ, sd.flux)

    # The viscosity parameter
    ν = get_gas_law(sd)[3]
    
    # add the fluxes due to the actual gas law
    @fastmath @inbounds Threads.@threads for i in eachindex(fxμy)
        pressure = p(ρ[i], sd)
        # Computing the momentum fluxes. 
        fxμx[i] += μx[i] * μx[i] / ρ[i] + pressure;        fxμy[i] += μx[i] .* μy[i] ./ ρ[i]
        fyμx[i] += μy[i] * μx[i] / ρ[i];                   fyμy[i] += μy[i] .* μy[i] ./ ρ[i] + pressure 
        # Computing the mass fluxes.
        fxρ[i]  += μx[i];                                  fyρ[i]  += μy[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m, n = get_size(sd)
        Δx, Δy = get_Δs(sd)
        for i in 1 : m, j = 1 : n
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1
            
            ∂xux = (μx[iu, j] / ρ[iu, j] - μx[id, j] / ρ[id, j]) / 2 / Δx
            ∂xuy = (μy[iu, j] / ρ[iu, j] - μy[id, j] / ρ[id, j]) / 2 / Δx

            ∂yux = (μx[i, ju] / ρ[i, ju] - μx[i, jd] / ρ[i, jd]) / 2 / Δy
            ∂yuy = (μy[i, ju] / ρ[i, ju] - μy[i, jd] / ρ[i, jd]) / 2 / Δy

            fxμx[i, j] -= ν * (∂xux + ∂xux) / 2
            fyμy[i, j] -= ν * (∂yuy + ∂yuy) / 2
            fxμy[i, j] -= ν * (∂xuy + ∂yux) / 2
            fyμx[i, j] -= ν * (∂yux + ∂xuy) / 2
        end
    end
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μx_out, μy_out, ρ_out, sd::SemidiscretizationTRLWIGR,
                  μx_old, μy_old, ρ_old)
    m, n = get_size(sd)
    Δx, Δy, Δt = get_Δs(sd)
    fxμx = sd.fxμx 
    fyμx = sd.fyμx 
    fxμy = sd.fxμy 
    fyμy = sd.fyμy 
    fxρ  = sd.fxρ  
    fyρ  = sd.fyρ  
    μx_inter = sd.μx_inter
    μy_inter = sd.μy_inter
    ρ_inter = sd.ρ_inter

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μx_old, μy_old, ρ_old)

    Threads.@threads for j = 1 : n 
        @fastmath @inbounds @simd for i = 1 : m
            # Compute the indices before and after, accounting for periodicity
            # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
            # Accordingly, id = i, iu = u etc, since they correspond to half-step
            # decrements of from the intermediate index i positioned in x[i + 1/2] 
            id = i
            iu = mod(i, m) + 1
            jd = j
            ju = mod(j, n) + 1

            μx_inter[i, j] = ((μx_old[iu, ju] + μx_old[iu, jd] + μx_old[id, ju] + μx_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxμx[iu, jd] + fxμx[iu, ju] - fxμx[id, jd] - fxμx[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyμx[id, ju] + fyμx[iu, ju] - fyμx[id, jd] - fyμx[iu, jd]))

            μy_inter[i, j] = ((μy_old[iu, ju] + μy_old[iu, jd] + μy_old[id, ju] + μy_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxμy[iu, jd] + fxμy[iu, ju] - fxμy[id, jd] - fxμy[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyμy[id, ju] + fyμy[iu, ju] - fyμy[id, jd] - fyμy[iu, jd]))

            ρ_inter[i, j]  = ((ρ_old[iu, ju] + ρ_old[iu, jd] + ρ_old[id, ju] + ρ_old[id, jd]) / 4 
                            - (Δt / (4 * Δx)) * (fxρ[iu, jd] + fxρ[iu, ju] - fxρ[id, jd] - fxρ[id, ju]) 
                            - (Δt / (4 * Δy)) * (fyρ[id, ju] + fyρ[iu, ju] - fyρ[id, jd] - fyρ[iu, jd]))
        end
    end

    # Compute the fluxes at the predictor point
    compute_fluxes!(sd, μx_inter, μy_inter, ρ_inter)

    Threads.@threads for j = 1 : n
        @fastmath @inbounds @simd  for i = 1 : m
            # Compute the indices before and after, accounting for periodicity
            # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
            # The indices for the fluxes around a location [i, j] in the primal mesh are
            # thus given as below. In the below, we are now re-using the *_new variables
            # as primal points 
            id = mod(i - 2, m) + 1
            iu = i
            jd = mod(j - 2, n) + 1
            ju = j

            # We are not multiplying with Δt, since this is done by the time stepper
            μx_out[i, j] = (  μx_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμx[iu, jd] + fxμx[iu, ju] - fxμx[id, jd] - fxμx[id, ju]) 
                            - (1 / (2 * Δy)) * (fyμx[id, ju] + fyμx[iu, ju] - fyμx[id, jd] - fyμx[iu, jd]))

            μy_out[i, j] = (  μy_out[i, j] 
                            - (1 / (2 * Δx)) * (fxμy[iu, jd] + fxμy[iu, ju] - fxμy[id, jd] - fxμy[id, ju]) 
                            - (1 / (2 * Δy)) * (fyμy[id, ju] + fyμy[iu, ju] - fyμy[id, jd] - fyμy[iu, jd]))

            ρ_out[i, j]  = (  ρ_out[i, j] 
                            - (1 / (2 * Δx)) * (fxρ[iu, jd] + fxρ[iu, ju] - fxρ[id, jd] - fxρ[id, ju]) 
                            - (1 / (2 * Δy)) * (fyρ[id, ju] + fyρ[iu, ju] - fyρ[id, jd] - fyρ[iu, jd]))
       end
    end
end
