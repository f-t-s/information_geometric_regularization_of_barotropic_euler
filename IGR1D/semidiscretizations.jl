using SparseArrays
using LinearAlgebra
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

"""
    Richtmyer Lax-Wendroff Semidiscretization
"""
struct SemidiscretizationRLW<:AbstractBarotropicSemidiscretization
    # We compute the pressure as P(ρ) = a ρ^γ
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float

    # The spatial discretization size
    Δx::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Wendroff method 
    fμ::Vector{Float}
    fρ::Vector{Float}

    # Storage for the intermediate step of TRLW 
    μ_inter::Vector{Float}
    ρ_inter::Vector{Float}
end

"""
    Constructor for TRLW semidiscretization 
"""
function SemidiscretizationRLW(a, γ, ν, Δx, Δt, m)
    return SemidiscretizationRLW(a, γ, ν, Δx, Δt,
                                 zeros(Float, m),
                                 zeros(Float, m),
                                 zeros(Float, m),
                                 zeros(Float, m))
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
    return size(sd.fμ)
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
    return sd.Δx, sd.Δt
end

"""
    This function computes the fluxes (of sd) in place.
"""
function compute_fluxes!(sd::AbstractBarotropicSemidiscretization, μ, ρ)
    # Ensures that all fluxes have the correct size
    fμ = sd.fμ 
    fρ = sd.fρ  
    m = get_size(sd)[1]

    # The viscosity parameter
    ν = get_gas_law(sd)[3]
    
    @fastmath @inbounds Threads.@threads for i in eachindex(fμ)
    # for i in eachindex(fμ)
        # Computing the momentum flux. 
        fμ[i] = μ[i] * μ[i] / ρ[i] + p(ρ[i], sd)
        # Computing the mass flux
        fρ[i]  = μ[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m = get_size(sd)[1]
        Δx = get_Δs(sd)[1]
        for i in 1 : m
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            
            ∂xu = (μ[iu] / ρ[iu] - μ[id] / ρ[id]) / 2 / Δx

            fμ[i] -= ν * ∂xu
        end
    end
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationRLW,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    Δx, Δt = get_Δs(sd)
    fμ = sd.fμ 
    fρ  = sd.fρ  
    μ_inter = sd.μ_inter
    ρ_inter = sd.ρ_inter

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μ_old, ρ_old)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2] 
        id = i
        iu = mod(i, m) + 1

        μ_inter[i] = (μ_old[iu] + μ_old[id]) / 2 - (Δt / (2 * Δx)) * (fμ[iu] - fμ[id])
        ρ_inter[i] = (ρ_old[iu] + ρ_old[id]) / 2 - (Δt / (2 * Δx)) * (fρ[iu] - fρ[id])
    end

    # Compute the fluxes at the predictor point
    compute_fluxes!(sd, μ_inter, ρ_inter)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
        # The indices for the fluxes around a location [i, j] in the primal mesh are
        # thus given as below. In the below, we are now re-using the *_new variables
        # as primal points 
        id = mod(i - 2, m) + 1
        iu = i

        # We are not multiplying with Δt, since this is done by the time stepper
        μ_out[i] = μ_out[i] - (1 / Δx) * (fμ[iu] - fμ[id])
        ρ_out[i] = ρ_out[i] - (1 / Δx) * (fρ[iu] - fρ[id])
    end
end

struct DoubleSemidiscretization
    sd1
    sd2
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::DoubleSemidiscretization, μ_old, ρ_old)
    add_rhs!(μ_out, ρ_out, sd.sd1, μ_old, ρ_old)
    add_rhs!(μ_out, ρ_out, sd.sd2, μ_old, ρ_old)
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
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Friedrichs method 
    fμ::Vector{Float}
    fρ::Vector{Float}
end

"""
    Constructor for LF semidiscretization that creates the scratch
    spaces automatically 
"""
function SemidiscretizationLF(a, γ, ν, Δx, Δt, m)
    return SemidiscretizationLF(a, γ, ν, Δx, Δt,
                                  zeros(Float, m),
                                  zeros(Float, m))
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationLF,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    Δx, Δt = get_Δs(sd)
    fμ = sd.fμ 
    fρ = sd.fρ  

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μ_old, ρ_old)

    @fastmath @inbounds Threads.@threads for i = 1 : m
    # for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2] 
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1

        μ_out[i] += ((μ_old[iu] + μ_old[id]) / 2 - μ_old[i]) / Δt - (1 / (2 * Δx)) * (fμ[iu] - fμ[id])
        ρ_out[i]  += ((ρ_old[iu] + ρ_old[id]) / 2 - ρ_old[i]) / Δt - (1 / (2 * Δx)) * (fρ[iu] - fρ[id])
    end
end

struct SemidiscretizationIGR<:AbstractSemidiscretization
    # The weight of the regularization 
    α::Float

    # The spatial discretization size
    Δx::Float

    # The matrix holding the elliptic problem
    elliptic_problem::SparseMatrixCSC{Float, Int}

    # The scratch space for holding the factorized LU problem
    elliptic_factorization::SparseArrays.UMFPACK.UmfpackLU{Float, Int}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_size(sd::SemidiscretizationIGR)
    return size(sd.Σ)
end


# assembles the sparsity pattern of the problem 
# it uses nonzero values corresponding to $1 = Δx = Δy = α ≡ ρ$
function assemble_elliptic_problem(m)
    I = Int[]
    J = Int[]
    V = Float[]

    for i = 1 : m
        iu = mod(i, m) + 1

        # Add the diagonal part (of the regularization)
        push!(I, i); push!(J, i); push!(V, 1.0)

        # Add the entries corresponding to the edge from i to iu
        push!(I, i); push!(J, i); push!(V, 1.0)

        push!(I, iu); push!(J, i); push!(V, -1.0)

        push!(I, i); push!(J, iu); push!(V, -1.0)

        push!(I, iu); push!(J, iu); push!(V, 1.0)
    end
    return sparse(I, J, V)
end

function SemidiscretizationIGR(α, Δx, m)
    elliptic_problem = assemble_elliptic_problem(m)
    elliptic_factorization = lu(elliptic_problem)
    Σ = zeros(m)
    return SemidiscretizationIGR(α, Δx, elliptic_problem, elliptic_factorization, Σ)
end

# We update the elliptic problem for the current density ρ
function update_elliptic_problem!(sd, ρ)
    α = sd.α
    ep = sd.elliptic_problem
    m = get_size(sd)[1]
    Δx =sd.Δx

    # going column by column
    for j = 1 : m
        # looking at all structural nonzeros of that column
        for ptr = ep.colptr[j] : (ep.colptr[j + 1] - 1) 
            # compute the row of the current entry considered 
            i = ep.rowval[ptr]
            
            # The nearest i coordinates to pick the right 
            iu = mod(i, m) + 1
            id = mod(i - 2, m) + 1

            if i == j # If this is a diagonal entry
                ep.nzval[ptr] = 1 / ρ[i] + α * (2 / ((ρ[i] + ρ[iu]) * Δx ^ 2) + 2 / ((ρ[i] + ρ[id]) * Δx ^ 2))
            elseif iu == j #If it's a right difference
                ep.nzval[ptr] = - α * 2 / ((ρ[i] + ρ[iu]) * Δx ^ 2) 
            elseif id == j # If it's a left difference 
                ep.nzval[ptr] = - α * 2 / ((ρ[i] + ρ[id]) * Δx ^ 2) 
            else
                throw(error("Not a valid entry of finite difference Laplacian"))
            end
        end
    end
end

# 
function solve_elliptic_problem!(sd)
    lu!(sd.elliptic_factorization, sd.elliptic_problem)
    ldiv!(sd.elliptic_factorization, sd.Σ)
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationIGR,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    α = sd.α
    Δx = sd.Δx
    Σ = sd.Σ

    # compute the IGR flux

    # computing the two Du squared terms
    # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        iu = mod(i, m) + 1
        id = mod(i - 2, m) + 1
        Σ[i] = 2 * α * ((μ_old[iu] ./ ρ_old[iu] - μ_old[id] ./ ρ_old[id]) / 2 / Δx) ^ 2
    end

    # update the elliptic problem
    update_elliptic_problem!(sd, ρ_old)

    # solve the elliptic problem
    solve_elliptic_problem!(sd)

    # using fluxes via central differences to update the state
    # @fastmath @inbounds @simd for i = 1 : m 
    for i = 1 : m 
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        μ_out[i] = (μ_out[i] - (1 / (2 * Δx)) * (Σ[iu] - Σ[id]))
    end
end

"""
    Semidiscretization for Saint-Venant Regularization, following Guelmame et al.
    Amounts to dividing the entropy pressure by 2. Only follows SVR exactly in the
    special case of γ = 1 (ideal gas).
"""
struct SemidiscretizationSVR<:AbstractSemidiscretization
    # The weight of the regularization 
    α::Float

    # The spatial discretization size
    Δx::Float

    # The matrix holding the elliptic problem
    elliptic_problem::SparseMatrixCSC{Float, Int}

    # The scratch space for holding the factorized LU problem
    elliptic_factorization::SparseArrays.UMFPACK.UmfpackLU{Float, Int}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_size(sd::SemidiscretizationSVR)
    return size(sd.Σ)
end

function SemidiscretizationSVR(α, Δx, m)
    elliptic_problem = assemble_elliptic_problem(m)
    elliptic_factorization = lu(elliptic_problem)
    Σ = zeros(m)
    return SemidiscretizationSVR(α, Δx, elliptic_problem, elliptic_factorization, Σ)
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationSVR,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    α = sd.α
    Δx = sd.Δx
    Σ = sd.Σ

    # compute the IGR flux

    # computing the two Du squared terms
    # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        iu = mod(i, m) + 1
        id = mod(i - 2, m) + 1
        Σ[i] = α * ((μ_old[iu] ./ ρ_old[iu] - μ_old[id] ./ ρ_old[id]) / 2 / Δx) ^ 2
    end

    # update the elliptic problem
    update_elliptic_problem!(sd, ρ_old)

    # solve the elliptic problem
    solve_elliptic_problem!(sd)

    # using fluxes via central differences to update the state
    # @fastmath @inbounds @simd for i = 1 : m 
    for i = 1 : m 
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        μ_out[i] = (μ_out[i] - (1 / (2 * Δx)) * (Σ[iu] - Σ[id]))
    end
end



"""
    Semidiscretization for Central Differences
"""
struct SemidiscretizationCD<:AbstractBarotropicSemidiscretization
    # We compute the pressure as P(ρ) = a ρ^γ
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float

    # The spatial discretization sizes
    Δx::Float
    # The temporal discretization size (needed for )
    Δt::Float

    # Storage for the fluxes computed during the Lax Friedrichs method 
    fμ::Vector{Float}
    fρ::Vector{Float}
end

"""
    Constructor for LF semidiscretization that creates the scratch
    spaces automatically 
"""
function SemidiscretizationCD(a, γ, ν, Δx, Δt, m)
    return SemidiscretizationCD(a, γ, ν, Δx, Δt,
                                  zeros(Float, m),
                                  zeros(Float, m))
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationCD,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    Δx, Δt = get_Δs(sd)
    fμ = sd.fμ 
    fρ = sd.fρ  

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μ_old, ρ_old)

        # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2] 
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1

        μ_out[i] -= (1 / (2 * Δx)) * (fμ[iu] - fμ[id])
        ρ_out[i] -= (1 / (2 * Δx)) * (fρ[iu] - fρ[id])
    end
end

struct SemidiscretizationLAD<:AbstractSemidiscretization
    # The weight of the regularization 
    α::Float

    # The spatial discretization size
    Δx::Float

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

"""
    This function returns the size of the grid of a semidiscretization
"""
function get_size(sd::SemidiscretizationLAD)
    return size(sd.Σ)
end

function SemidiscretizationLAD(α, Δx, m)
    Σ = zeros(m)
    return SemidiscretizationLAD(α, Δx, Σ)
end

"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationLAD,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    α = sd.α
    Δx = sd.Δx
    Σ = sd.Σ

    # compute the IGR flux

    # computing the two Du squared terms
    # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        iu = mod(i, m) + 1
        id = mod(i - 2, m) + 1
        if (μ_old[iu] ./ ρ_old[iu] - μ_old[id] ./ ρ_old[id]) < 0
            Σ[i] = 2 * α * ((μ_old[iu] ./ ρ_old[iu] - μ_old[id] ./ ρ_old[id]) / 2 / Δx) ^ 2
        else 
            Σ[i] = 0.0
        end
    end

    # using fluxes via central differences to update the state
    # @fastmath @inbounds @simd for i = 1 : m 
    for i = 1 : m 
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        μ_out[i] = (μ_out[i] - (1 / (2 * Δx)) * (Σ[iu] - Σ[id]))
    end
end

struct SemidiscretizationRLWIGR<:AbstractBarotropicSemidiscretization
    # The gas parameters
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float
    # The weight of the regularization 
    α::Float

    # The spatial discretization size
    Δx::Float
    Δt::Float

    # Storage for the fluxes computed during the Lax Wendroff method 
    fμ::Vector{Float}
    fρ::Vector{Float}

    # Storage for the intermediate step of TRLW 
    μ_inter::Vector{Float}
    ρ_inter::Vector{Float}



    # The matrix holding the elliptic problem
    elliptic_problem::SparseMatrixCSC{Float, Int}

    # The scratch space for holding the factorized LU problem
    elliptic_factorization::SparseArrays.UMFPACK.UmfpackLU{Float, Int}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

function SemidiscretizationRLWIGR(a, γ, ν, α, Δx, Δt, m)
    elliptic_problem = assemble_elliptic_problem(m)
    elliptic_factorization = lu(elliptic_problem)
    Σ = zeros(m)
    return SemidiscretizationRLWIGR(a, γ, ν, α, Δx, Δt, 
                                    zeros(Float, m),
                                    zeros(Float, m),
                                    zeros(Float, m),
                                    zeros(Float, m), 
                                    elliptic_problem, elliptic_factorization, Σ)
end


"""
    This function computes the fluxes (of sd) in place.
"""
function compute_fluxes!(sd::SemidiscretizationRLWIGR, μ, ρ)
    # binds the fluxes of sd to variables
    fμ = sd.fμ 
    fρ = sd.fρ  
    Δx = get_Δs(sd)[1]
    m = get_size(sd)[1]

    # The viscosity parameter
    ν = get_gas_law(sd)[3]
    
    @fastmath @inbounds Threads.@threads for i in eachindex(fμ)
    # for i in eachindex(fμ)
        # Computing the momentum flux. 
        fμ[i] = μ[i] * μ[i] / ρ[i] + p(ρ[i], sd)
        # Computing the mass flux
        fρ[i]  = μ[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m = get_size(sd)[1]
        for i in 1 : m
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            
            ∂xu = (μ[iu] / ρ[iu] - μ[id] / ρ[id]) / 2 / Δx

            fμ[i] -= ν * ∂xu
        end
    end

    α = sd.α
    Σ = sd.Σ

    # compute the IGR flux

    # computing the two Du squared terms
    # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        iu = mod(i, m) + 1
        id = mod(i - 2, m) + 1
        Σ[i] = 2 * α * ((μ[iu] ./ ρ[iu] - μ[id] ./ ρ[id]) / 2 / Δx) ^ 2
    end

    # update the elliptic problem
    update_elliptic_problem!(sd, ρ)

    # solve the elliptic problem
    solve_elliptic_problem!(sd)

    for i = 1 : m
        fμ[i] += Σ[i]
    end
end
"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationRLWIGR,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    Δx, Δt = get_Δs(sd)
    fμ = sd.fμ 
    fρ  = sd.fρ  
    μ_inter = sd.μ_inter
    ρ_inter = sd.ρ_inter

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μ_old, ρ_old)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2] 
        id = i
        iu = mod(i, m) + 1

        μ_inter[i] = (μ_old[iu] + μ_old[id]) / 2 - (Δt / (2 * Δx)) * (fμ[iu] - fμ[id])
        ρ_inter[i] = (ρ_old[iu] + ρ_old[id]) / 2 - (Δt / (2 * Δx)) * (fρ[iu] - fρ[id])
    end

    # Compute the fluxes at the predictor point
    compute_fluxes!(sd, μ_inter, ρ_inter)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
        # The indices for the fluxes around a location [i, j] in the primal mesh are
        # thus given as below. In the below, we are now re-using the *_new variables
        # as primal points 
        id = mod(i - 2, m) + 1
        iu = i

        # We are not multiplying with Δt, since this is done by the time stepper
        μ_out[i] = μ_out[i] - (1 / Δx) * (fμ[iu] - fμ[id])
        ρ_out[i] = ρ_out[i] - (1 / Δx) * (fρ[iu] - fρ[id])
    end
end

struct SemidiscretizationRLWLAD<:AbstractBarotropicSemidiscretization
    # The gas parameters
    a::Float
    γ::Float

    # The viscosity parameter
    ν::Float
    # The weight of the regularization 
    α::Float

    # The spatial discretization size
    Δx::Float
    Δt::Float

    # Storage for the fluxes computed during the Lax Wendroff method 
    fμ::Vector{Float}
    fρ::Vector{Float}

    # Storage for the intermediate step of TRLW 
    μ_inter::Vector{Float}
    ρ_inter::Vector{Float}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

function SemidiscretizationRLWLAD(a, γ, ν, α, Δx, Δt, m)
    Σ = zeros(m)
    return SemidiscretizationRLWLAD(a, γ, ν, α, Δx, Δt, 
                                    zeros(Float, m),
                                    zeros(Float, m),
                                    zeros(Float, m),
                                    zeros(Float, m), 
                                    Σ)
end


"""
    This function computes the fluxes (of sd) in place.
"""
function compute_fluxes!(sd::SemidiscretizationRLWLAD, μ, ρ)
    # binds the fluxes of sd to variables
    fμ = sd.fμ 
    fρ = sd.fρ  
    Δx = get_Δs(sd)[1]
    m = get_size(sd)[1]

    # The viscosity parameter
    ν = get_gas_law(sd)[3]
    
    @fastmath @inbounds Threads.@threads for i in eachindex(fμ)
    # for i in eachindex(fμ)
        # Computing the momentum flux. 
        fμ[i] = μ[i] * μ[i] / ρ[i] + p(ρ[i], sd)
        # Computing the mass flux
        fρ[i]  = μ[i]
    end

    # Computing the viscosity
    if ν ≠ 0 
        m = get_size(sd)[1]
        for i in 1 : m
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            
            ∂xu = (μ[iu] / ρ[iu] - μ[id] / ρ[id]) / 2 / Δx

            fμ[i] -= ν * ∂xu
        end
    end

    α = sd.α
    Σ = sd.Σ

    # compute the IGR flux

    # computing the two Du squared terms
    # @fastmath @inbounds @simd for i = 1 : m
    for i = 1 : m
        iu = mod(i, m) + 1
        id = mod(i - 2, m) + 1
        if (μ[iu] ./ ρ[iu] - μ[id] ./ ρ[id]) < 0
            Σ[i] = 2 * α * ((μ[iu] ./ ρ[iu] - μ[id] ./ ρ[id]) / 2 / Δx) ^ 2
        else 
            Σ[i] = 0.0
        end
    end

    for i = 1 : m
        fμ[i] += Σ[i]
    end
end
"""
    Adds f(x_old) to x_out. Modifies all arguments up to and including sd
"""
function add_rhs!(μ_out, ρ_out, sd::SemidiscretizationRLWLAD,
                  μ_old, ρ_old)
    m = get_size(sd)[1]
    Δx, Δt = get_Δs(sd)
    fμ = sd.fμ 
    fρ  = sd.fρ  
    μ_inter = sd.μ_inter
    ρ_inter = sd.ρ_inter

    # Compute the fluxes in starting point
    compute_fluxes!(sd, μ_old, ρ_old)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Here, u_new[i, j] corresponds to the location x[i + 1/2], y[i + 1/2]
        # Accordingly, id = i, iu = u etc, since they correspond to half-step
        # decrements of from the intermediate index i positioned in x[i + 1/2] 
        id = i
        iu = mod(i, m) + 1

        μ_inter[i] = (μ_old[iu] + μ_old[id]) / 2 - (Δt / (2 * Δx)) * (fμ[iu] - fμ[id])
        ρ_inter[i] = (ρ_old[iu] + ρ_old[id]) / 2 - (Δt / (2 * Δx)) * (fρ[iu] - fρ[id])
    end

    # Compute the fluxes at the predictor point
    compute_fluxes!(sd, μ_inter, ρ_inter)

    # @fastmath @inbounds @simd  for i = 1 : m
    for i = 1 : m
        # Compute the indices before and after, accounting for periodicity
        # Now, flux[i, j] corresponds to the flux in location x[i + 1/2], y[i + 1/2]
        # The indices for the fluxes around a location [i, j] in the primal mesh are
        # thus given as below. In the below, we are now re-using the *_new variables
        # as primal points 
        id = mod(i - 2, m) + 1
        iu = i

        # We are not multiplying with Δt, since this is done by the time stepper
        μ_out[i] = μ_out[i] - (1 / Δx) * (fμ[iu] - fμ[id])
        ρ_out[i] = ρ_out[i] - (1 / Δx) * (fρ[iu] - fρ[id])
    end
end

# computes the potential, kinetic, and total energies
function compute_energies(sd, u, ρ)
    nx = size(ρ, 1)
    nt = size(ρ, 2)
    kinetic_energies = [sum(u[:, i].^2  .* ρ[:, i]) / 2 for i = 1 : nt] / nx
    # Computing the potential energy at each time step
    potential_energies = [sum(e.(ρ[:, i], [sd]) .* ρ[:, i]) for i = 1 : nt] / nx 
    equilibrium_energy = sum(ρ[:, 1]) * e(sum(ρ[:, 1]) / length(ρ[:, 1]), sd) / nx 
    total_energies = kinetic_energies + potential_energies
    return kinetic_energies, potential_energies, equilibrium_energy, total_energies
end