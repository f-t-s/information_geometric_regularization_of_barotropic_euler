"""
  function for computing the components of Du using regularized central differences
"""
function compute_Du_reg_central!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
    Threads.@threads for i = 1 : m 
        @fastmath @inbounds @simd for j = 1 : n
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1

            ∂xux[i, j] = ((μx[iu, jd] / ρ[iu, jd] + 2 * μx[iu, j] / ρ[iu, j] + μx[iu, ju] / ρ[iu, ju]) 
                         -(μx[id, jd] / ρ[id, jd] + 2 * μx[id, j] / ρ[id, j] + μx[id, ju] / ρ[id, ju])) / 8 / Δx
            ∂xuy[i, j] = ((μy[iu, jd] / ρ[iu, jd] + 2 * μy[iu, j] / ρ[iu, j] + μy[iu, ju] / ρ[iu, ju]) 
                         -(μy[id, jd] / ρ[id, jd] + 2 * μy[id, j] / ρ[id, j] + μy[id, ju] / ρ[id, ju])) / 8 / Δx
            ∂yux[i, j] = ((μx[id, ju] / ρ[id, ju] + 2 * μx[i, ju] / ρ[i, ju] + μx[iu, ju] / ρ[iu, ju]) 
                         -(μx[id, jd] / ρ[id, jd] + 2 * μx[i, jd] / ρ[i, jd] + μx[iu, jd] / ρ[iu, jd])) / 8 / Δy
            ∂yuy[i, j] = ((μy[id, ju] / ρ[id, ju] + 2 * μy[i, ju] / ρ[i, ju] + μy[iu, ju] / ρ[iu, ju]) 
                         -(μy[id, jd] / ρ[id, jd] + 2 * μy[i, jd] / ρ[i, jd] + μy[iu, jd] / ρ[iu, jd])) / 8 / Δy
        end
    end
end

function compute_Du_reg_left!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
    for i = 1 : m, j = 1 : n
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        jd = mod(j - 2, n) + 1
        ju = mod(j, n) + 1

        ∂xux[i, j] = ((μx[i, jd]  / ρ[i, jd]  + 2 * μx[i, j]  / ρ[i, j]  + μx[i, ju]  / ρ[i, ju]) 
                     -(μx[id, jd] / ρ[id, jd] + 2 * μx[id, j] / ρ[id, j] + μx[id, ju] / ρ[id, ju])) / 4 / Δx
        ∂xuy[i, j] = ((μy[i, jd]  / ρ[i, jd]  + 2 * μy[i, j]  / ρ[i, j]  + μy[i, ju]  / ρ[i, ju]) 
                     -(μy[id, jd] / ρ[id, jd] + 2 * μy[id, j] / ρ[id, j] + μy[id, ju] / ρ[id, ju])) / 4 / Δx
        ∂yux[i, j] = ((μx[id, j]  / ρ[id, j]  + 2 * μx[i, j]  / ρ[i, j]  + μx[iu, j]  / ρ[iu, j]) 
                     -(μx[id, jd] / ρ[id, jd] + 2 * μx[i, jd] / ρ[i, jd] + μx[iu, jd] / ρ[iu, jd])) / 4 / Δy
        ∂yuy[i, j] = ((μy[id, j]  / ρ[id, j]  + 2 * μy[i, j]  / ρ[i, j]  + μy[iu, j]  / ρ[iu, j]) 
                     -(μy[id, jd] / ρ[id, jd] + 2 * μy[i, jd] / ρ[i, jd] + μy[iu, jd] / ρ[iu, jd])) / 4 / Δy
    end
end

function compute_Du_reg_right!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
    for i = 1 : m, j = 1 : n
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        jd = mod(j - 2, n) + 1
        ju = mod(j, n) + 1

        ∂xux[i, j] = ((μx[iu, jd] / ρ[iu, jd] + 2 * μx[iu, j] / ρ[iu, j] + μx[iu, ju] / ρ[iu, ju]) 
                     -(μx[i, jd]  / ρ[i, jd]  + 2 * μx[i, j]  / ρ[i, j]  + μx[i, ju]  / ρ[i, ju])) / 4 / Δx
        ∂xuy[i, j] = ((μy[iu, jd] / ρ[iu, jd] + 2 * μy[iu, j] / ρ[iu, j] + μy[iu, ju] / ρ[iu, ju]) 
                     -(μy[i, jd]  / ρ[i, jd]  + 2 * μy[i, j]  / ρ[i, j]  + μy[i, ju]  / ρ[i, ju])) / 4 / Δx
        ∂yux[i, j] = ((μx[id, ju] / ρ[id, ju] + 2 * μx[i, ju] / ρ[i, ju] + μx[iu, ju] / ρ[iu, ju]) 
                     -(μx[id, j]  / ρ[id, j]  + 2 * μx[i, j]  / ρ[i, j]  + μx[iu, j]  / ρ[iu, j])) / 4 / Δy
        ∂yuy[i, j] = ((μy[id, ju] / ρ[id, ju] + 2 * μy[i, ju] / ρ[i, ju] + μy[iu, ju] / ρ[iu, ju]) 
                     -(μy[id, j]  / ρ[id, j]  + 2 * μy[i, j]  / ρ[i, j]  + μy[iu, j]  / ρ[iu, j])) / 4 / Δy
    end
end

# function for computing the components of Du using central diff
function compute_Du_central!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
        Threads.@threads for  j = 1 : n
            @fastmath @inbounds @simd for i = 1 : m
            id = mod(i - 2, m) + 1
            iu = mod(i, m) + 1
            jd = mod(j - 2, n) + 1
            ju = mod(j, n) + 1

            ∂xux[i, j] = (μx[iu, j] / ρ[iu, j] - μx[id, j] / ρ[id, j]) / 2 / Δx
            ∂xuy[i, j] = (μy[iu, j] / ρ[iu, j] - μy[id, j] / ρ[id, j]) / 2 / Δx
            ∂yux[i, j] = (μx[i, ju] / ρ[i, ju] - μx[i, jd] / ρ[i, jd]) / 2 / Δy
            ∂yuy[i, j] = (μy[i, ju] / ρ[i, ju] - μy[i, jd] / ρ[i, jd]) / 2 / Δy
        end
    end
end


# function for computing the components of Du using left finite differences
function compute_Du_left!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
    for i = 1 : m, j = 1 : n
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        jd = mod(j - 2, n) + 1
        ju = mod(j, n) + 1

        ∂xux[i, j] = ((μx[i, j] / ρ[i, j] - μx[id, j] / ρ[id, j])) / Δx
        ∂xuy[i, j] = ((μy[i, j] / ρ[i, j] - μy[id, j] / ρ[id, j])) / Δx
        ∂yux[i, j] = ((μx[i, j] / ρ[i, j] - μx[i, jd] / ρ[i, jd])) / Δy
        ∂yuy[i, j] = ((μy[i, j] / ρ[i, j] - μy[i, jd] / ρ[i, jd])) / Δy
    end
end

# function for computing the components of Du using right finite differences
function compute_Du_right!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
    for i = 1 : m, j = 1 : n
        id = mod(i - 2, m) + 1
        iu = mod(i, m) + 1
        jd = mod(j - 2, n) + 1
        ju = mod(j, n) + 1

        ∂xux[i, j] = ((μx[iu, j] / ρ[iu, j] - μx[i, j] / ρ[i, j])) / Δx
        ∂xuy[i, j] = ((μy[iu, j] / ρ[iu, j] - μy[i, j] / ρ[i, j])) / Δx
        ∂yux[i, j] = ((μx[i, ju] / ρ[i, ju] - μx[i, j] / ρ[i, j])) / Δy
        ∂yuy[i, j] = ((μy[i, ju] / ρ[i, ju] - μy[i, j] / ρ[i, j])) / Δy
    end
end

# function for computing the components of Du using central diff
function compute_Du_staggered!(∂xux, ∂xuy, ∂yux, ∂yuy ,μx, μy, ρ, Δx, Δy)
    m, n = size(μx)
        Threads.@threads for  j = 1 : n
            @fastmath @inbounds @simd for i = 1 : m
            id = i
            iu = mod(i, m) + 1
            jd = j
            ju = mod(j, n) + 1

            ∂xux[i, j] = (μx[iu, ju] / ρ[iu, ju] - μx[id, ju] / ρ[id, ju] + μx[iu, jd] / ρ[iu, jd] - μx[id, jd] / ρ[id, jd]) / 2 / Δx
            ∂xuy[i, j] = (μy[iu, ju] / ρ[iu, ju] - μy[id, ju] / ρ[id, ju] + μy[iu, jd] / ρ[iu, jd] - μy[id, jd] / ρ[id, jd]) / 2 / Δx
            ∂yux[i, j] = (μx[iu, ju] / ρ[iu, ju] - μx[iu, jd] / ρ[iu, jd] + μx[id, ju] / ρ[id, ju] - μx[id, jd] / ρ[id, jd]) / 2 / Δy
            ∂yuy[i, j] = (μy[iu, ju] / ρ[iu, ju] - μy[iu, jd] / ρ[iu, jd] + μy[id, ju] / ρ[id, ju] - μy[id, jd] / ρ[id, jd]) / 2 / Δy
        end
    end
end
