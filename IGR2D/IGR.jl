using SparseArrays
using LinearAlgebra
abstract type AbstractIGRFlux end

"""
    An IGR flux that uses LU factorization to solve for the entropic pressure
"""
struct IGRFluxLU<:AbstractIGRFlux
    α::Float
    Δx::Float
    Δy::Float
    m::Int
    n::Int

    # The matrix holding the elliptic problem
    elliptic_problem::SparseMatrixCSC{Float, Int}

    # The scratch space for holding the factorized LU problem
    elliptic_factorization::SparseArrays.UMFPACK.UmfpackLU{Float, Int}

    # pre-allocation of variables for Du
    ∂xux::Matrix{Float}
    ∂xuy::Matrix{Float}
    ∂yux::Matrix{Float}
    ∂yuy::Matrix{Float}
 
    # pre-allocation for the two scalar fields creating rhs
    trDu2::Matrix{Float}
    tr2Du::Matrix{Float}

    # Pre-allocation for rhs of elliptic problem
    rhs::Vector{Float}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

""" 
    Constructor for IGRFluxLU
"""
function IGRFluxLU(α, Δx, Δy, m, n)
    # The total number of dofs
    N = m * n

    # the elliptic operator 
    elliptic_problem = assemble_elliptic_problem(m, n)

    elliptic_factorization = lu(elliptic_problem)

    ∂xux = zeros(m, n)
    ∂xuy = zeros(m, n)
    ∂yux = zeros(m, n) 
    ∂yuy = zeros(m, n) 

    trDu2 = zeros(m, n)
    tr2Du = zeros(m, n)

    rhs  = zeros(N)
    Σ  = zeros(N)

    return IGRFluxLU(α, Δx, Δy, m, n, elliptic_problem, elliptic_factorization, ∂xux, ∂xuy, ∂yux, ∂yuy, trDu2, tr2Du, rhs, Σ)
end

function get_Δs(flux::AbstractIGRFlux)
    return flux.Δx, flux.Δy
end

function get_size(flux::AbstractIGRFlux)
    return flux.m, flux.n
end


# assembles the sparsity pattern of the problem 
# it uses nonzero values corresponding to $1 = Δx = Δy = α ≡ ρ$
function assemble_elliptic_problem(m, n)
    linds = LinearIndices((m, n))

    I = Int[]
    J = Int[]
    V = Float[]

    # Assemble the 1, 1 block
    for i = 1 : m, j = 1 : n
        # Since we are working with central difference squared, we need to consider
        # the next nearest neighbor
        iu = mod(i, m) + 1
        ju = mod(j, n) + 1

        # Add the diagonal part (of the regularization)
        push!(I, linds[i, j])
        push!(J, linds[i, j])
        push!(V, 1.0)

        # Add the entries corresponding to iu
        push!(I, linds[i, j])
        push!(J, linds[i, j])
        push!(V, 1.0)

        push!(I, linds[iu, j])
        push!(J, linds[i , j])
        push!(V, -1.0)

        push!(I, linds[i , j])
        push!(J, linds[iu, j])
        push!(V, -1.0)

        push!(I, linds[iu, j])
        push!(J, linds[iu, j])
        push!(V, 1.0)

        # Add the entries corresponding to ju
        push!(I, linds[i, j])
        push!(J, linds[i, j])
        push!(V, 1.0)

        push!(I, linds[i, ju])
        push!(J, linds[i, j ])
        push!(V, -1.0)

        push!(I, linds[i, j ])
        push!(J, linds[i, ju])
        push!(V, -1.0)

        push!(I, linds[i, ju])
        push!(J, linds[i, ju])
        push!(V, 1.0)
    end
    @assert length(I) == length(J) == length(V) == 9 * m * n

    return sparse(I, J, V)
end

# update the matrix representation of the elliptic problem 
# used to solve for the entropic pressure Σ
function update_elliptic_problem!(flux::IGRFluxLU, ρ)
    α = flux.α
    ep = flux.elliptic_problem
    m, n = get_size(flux)
    N = m * n
    Δx, Δy = get_Δs(flux)
    cinds = CartesianIndices((m, n))
    linds = LinearIndices((m, n))

    # going column by column
    for j = 1 : N
        # looking at all structural nonzeros of that column
        for ptr = ep.colptr[j] : (ep.colptr[j + 1] - 1) 
            # compute the row of the current entry considered 
            i = ep.rowval[ptr]
            
            ix = cinds[i][1]; iy = cinds[i][2]
            jx = cinds[j][1]; jy = cinds[j][2]

            # The nearest ix and iy coordinates to pick the right xis
            ixu = mod(ix, m) + 1
            ixd = mod(ix - 2, m) + 1
            iyu = mod(iy, n) + 1
            iyd = mod(iy - 2, n) + 1

            if i == j # If this is a diagonal entry
                ep.nzval[ptr] = 1 / ρ[i] + α * (2 / ((ρ[linds[ix, iy]] + ρ[linds[ixu, iy]]) * Δx ^ 2) 
                                              + 2 / ((ρ[linds[ix, iy]] + ρ[linds[ixd, iy]]) * Δx ^ 2)
                                              + 2 / ((ρ[linds[ix, iy]] + ρ[linds[ix, iyu]]) * Δy ^ 2)
                                              + 2 / ((ρ[linds[ix, iy]] + ρ[linds[ix, iyd]]) * Δy ^ 2))
            elseif (iy == jy) # If this is an x-axis difference
                if ixu == jx #If it's a right difference
                    ep.nzval[ptr] = - α * 2 / ((ρ[ix, iy] + ρ[linds[ixu, iy]]) * Δx ^ 2) 
                elseif ixd == jx # If it's a left difference 
                    ep.nzval[ptr] = - α * 2 / ((ρ[ix, iy] + ρ[linds[ixd, iy]]) * Δx ^ 2) 
                else
                    throw(error("Not a valid entry of finite difference Laplacian"))
                end
            elseif (ix == jx)# If this is an x-axis difference
                if iyu == jy #If it's a right difference
                    ep.nzval[ptr] = - α * 2 / ((ρ[ix, iy] + ρ[linds[ix, iyu]]) * Δy ^ 2) 
                elseif iyd == jy # If it's a left difference 
                    ep.nzval[ptr] = - α * 2 / ((ρ[ix, iy] + ρ[linds[ix, iyd]]) * Δy ^ 2) 
                else
                    throw(error("Not a valid entry of finite difference Laplacian"))
                end
            else
                throw(error("Not a valid entry of finite difference Laplacian"))
            end
        end
    end
end

# Modifies the "reg" component of flux and overwrites it with the new regularization term
# computed with rhs. Uses ρ as current density.
function solve_elliptic_problem!(flux::IGRFluxLU, ρ)
    Σ = flux.Σ
    rhs = flux.rhs
    elliptic_problem = flux.elliptic_problem
    elliptic_factorization = flux.elliptic_factorization


    # updating the matrix used for the elliptic problem
    update_elliptic_problem!(flux, ρ)

    # update the LU factor
    lu!(elliptic_factorization, elliptic_problem)

    # solve the elliptic problem using the LU factorization
    ldiv!(Σ, elliptic_factorization, rhs)
end


"""
    An IGR flux that uses Gauss Seidel to solve for the entropic pressure
"""
struct IGRFluxGS<:AbstractIGRFlux
    α::Float
    max_iter::Int
    Δx::Float
    Δy::Float
    m::Int
    n::Int

    # pre-allocation of variables for Du
    ∂xux::Matrix{Float}
    ∂xuy::Matrix{Float}
    ∂yux::Matrix{Float}
    ∂yuy::Matrix{Float}
 
    # pre-allocation for the two scalar fields creating rhs
    trDu2::Matrix{Float}
    tr2Du::Matrix{Float}

    # Pre-allocation for rhs of elliptic problem
    rhs::Vector{Float}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}
end

""" 
    Constructor for IGRFluxGS
"""
function IGRFluxGS(α, max_iter, Δx, Δy, m, n)
    # The total number of dofs
    N = m * n

    ∂xux = zeros(m, n)
    ∂xuy = zeros(m, n)
    ∂yux = zeros(m, n) 
    ∂yuy = zeros(m, n) 

    trDu2 = zeros(m, n)
    tr2Du = zeros(m, n)

    rhs  = zeros(N)
    Σ  = zeros(N)

    return IGRFluxGS(α, max_iter, Δx, Δy, m, n, ∂xux, ∂xuy, ∂yux, ∂yuy, trDu2, tr2Du, rhs, Σ)
end

"""
    An IGR flux that uses Jacobi Iteration to solve for the entropic pressure
"""
struct IGRFluxJI<:AbstractIGRFlux
    α::Float
    max_iter::Int
    Δx::Float
    Δy::Float
    m::Int
    n::Int

    # pre-allocation of variables for Du
    ∂xux::Matrix{Float}
    ∂xuy::Matrix{Float}
    ∂yux::Matrix{Float}
    ∂yuy::Matrix{Float}
 
    # pre-allocation for the two scalar fields creating rhs
    trDu2::Matrix{Float}
    tr2Du::Matrix{Float}

    # Pre-allocation for rhs of elliptic problem
    rhs::Vector{Float}

    # Pre-allocation for the entropic pressure 
    Σ::Vector{Float}

    # Pre-allocation for the old pressure 
    Σ_old::Vector{Float}

end

""" 
    Constructor for IGRFluxJI
"""
function IGRFluxJI(α, max_iter, Δx, Δy, m, n)
    # The total number of dofs
    N = m * n

    ∂xux = zeros(m, n)
    ∂xuy = zeros(m, n)
    ∂yux = zeros(m, n) 
    ∂yuy = zeros(m, n) 

    trDu2 = zeros(m, n)
    tr2Du = zeros(m, n)

    rhs  = zeros(N)
    Σ  = zeros(N)
    Σ_old  = zeros(N)

    return IGRFluxJI(α, max_iter, Δx, Δy, m, n, ∂xux, ∂xuy, ∂yux, ∂yuy, trDu2, tr2Du, rhs, Σ, Σ_old)
end


"""
    Solving the elliptic subproblem via Gauss Seidel
"""
function solve_elliptic_problem!(flux::IGRFluxGS, ρ)
    α = flux.α
    m, n = get_size(flux)
    Δx, Δy = get_Δs(flux)
    Σ = reshape(flux.Σ, m, n)
    ρ = reshape(ρ, m, n)
    rhs = reshape(flux.rhs, m, n)
    max_iter = flux.max_iter

    for iter = 1 : max_iter
        # making the x loop the inner loop, due to column major
        # storage of ρ
        for j = 1 : n
            @fastmath @inbounds for i = 1 : m 
                # The nearest ix and iy coordinates to pick the right ρ
                iu = mod(i, m) + 1
                id = mod(i - 2, m) + 1
                ju = mod(j, n) + 1
                jd = mod(j - 2, n) + 1

                Σ[i, j] = (rhs[i, j] 
                              + 2 * α * Σ[id, j] / (ρ[i, j] + ρ[id, j]) / Δx ^ 2
                              + 2 * α * Σ[iu, j] / (ρ[i, j] + ρ[iu, j]) / Δx ^ 2
                              + 2 * α * Σ[i, jd] / (ρ[i, j] + ρ[i, jd]) / Δy ^ 2
                              + 2 * α * Σ[i, ju] / (ρ[i, j] + ρ[i, ju]) / Δy ^ 2) 
                Σ[i, j] /= (1 / ρ[i, j] + 2 * α / (ρ[i, j] + ρ[id, j]) / Δx ^ 2
                              + 2 * α / (ρ[i, j] + ρ[iu, j]) / Δx ^ 2
                              + 2 * α / (ρ[i, j] + ρ[i, jd]) / Δy ^ 2
                              + 2 * α / (ρ[i, j] + ρ[i, ju]) / Δy ^ 2)
            end
        end
    end
end

"""
    Solving the elliptic subproblem via Jacobi Iteration
"""
function solve_elliptic_problem!(flux::IGRFluxJI, ρ)
    # TODO: There is still a minor bug in that $Σ_old$ is not being updated
    α = flux.α
    m, n = get_size(flux)
    Δx, Δy = get_Δs(flux)
    Σ = reshape(flux.Σ, m, n)
    Σ_old = reshape(flux.Σ_old, m, n)
    ρ = reshape(ρ, m, n)
    rhs = reshape(flux.rhs, m, n)
    max_iter = flux.max_iter

    for iter = 1 : max_iter
        # making the x loop the inner loop, due to column major
        # storage of ρ
        Σ_old .= Σ
        Threads.@threads for j = 1 : n
            @fastmath @inbounds @simd for i = 1 : m 
                # The nearest ix and iy coordinates to pick the right ρ
                iu = mod(i, m) + 1
                id = mod(i - 2, m) + 1
                ju = mod(j, n) + 1
                jd = mod(j - 2, n) + 1

                Σ[i, j] = (rhs[i, j] 
                              + 2 * α * Σ_old[id, j] / (ρ[i, j] + ρ[id, j]) / Δx ^ 2
                              + 2 * α * Σ_old[iu, j] / (ρ[i, j] + ρ[iu, j]) / Δx ^ 2
                              + 2 * α * Σ_old[i, jd] / (ρ[i, j] + ρ[i, jd]) / Δy ^ 2
                              + 2 * α * Σ_old[i, ju] / (ρ[i, j] + ρ[i, ju]) / Δy ^ 2) 
                Σ[i, j] /= (1 / ρ[i, j] + 2 * α / (ρ[i, j] + ρ[id, j]) / Δx ^ 2
                              + 2 * α / (ρ[i, j] + ρ[iu, j]) / Δx ^ 2
                              + 2 * α / (ρ[i, j] + ρ[i, jd]) / Δy ^ 2
                              + 2 * α / (ρ[i, j] + ρ[i, ju]) / Δy ^ 2)
            end
        end
    end
end

function compute_fluxes!(fxμx, fyμx, fxμy, fyμy, fxρ, fyρ, μx, μy, ρ, flux::AbstractIGRFlux)
    # Importing variables from sd
    m = flux.m
    n = flux.n
    α     = flux.α
    ∂xux  = flux.∂xux
    ∂xuy  = flux.∂xuy
    ∂yux  = flux.∂yux
    ∂yuy  = flux.∂yuy
    rhs   = flux.rhs
    Σ     = flux.Σ

    # compute Du 
    compute_Du_reg_central!(∂xux, ∂xuy, ∂yux, ∂yuy, μx, μy, ρ, Δx, Δy)
    linds = LinearIndices((m, n))

    # computing the two Du squared terms
    Threads.@threads for j = 1 : n
        @fastmath @inbounds @simd for i = 1 : m
            trDu2 = (∂xux[i, j] * ∂xux[i, j] + ∂yux[i, j] * ∂xuy[i, j] 
                  +  ∂xuy[i, j] * ∂yux[i, j] + ∂yuy[i, j] * ∂yuy[i, j])
            tr2Du = (∂xux[i, j] * ∂xux[i, j] + ∂xux[i, j] * ∂yuy[i, j] 
                  +  ∂yuy[i, j] * ∂xux[i, j] + ∂yuy[i, j] * ∂yuy[i, j])
            rhs[linds[i, j]] = α * (trDu2 + tr2Du)
        end
    end

    # preparing the right hand side of the elliptic problem
    # rhs .= α .* vec(trDu2 .+ tr2Du)


    # solve the elliptic problem
    solve_elliptic_problem!(flux, ρ)

    # Adding the entropic pressure to the flux terms
    vec(fxμx) .= Σ
    vec(fxμy) .= 0.0
    vec(fyμx) .= 0.0
    vec(fyμy) .= Σ

    # IGR does not produce mass fluxes
    fxρ .= 0.0
    fyρ .= 0.0
end