"""
    Initializes an m by n sized problem on an Lx by Ly rectangular
    domain, with initial conditions given by the functions
    ux_funct, uy_funct, ρ_funct.
"""
function initialize(m, n, Lx, Ly, ux_funct, uy_funct, ρ_funct) 
    # Compute grid spacing
    Δx = Lx / m
    Δy = Ly / n
    
    # Compute the x and y coordinates we are adding a Δx / 2 and Δy / 2
    # so that the points are in the interior of [0, Lx] and [0, Ly]
    xs = Δx / 2 .+ (0 : (m - 1)) * Δx
    ys = Δy / 2 .+ (0 : (n - 1)) * Δy

    # Create arrays holding the initial conditions
    ux0 = zeros(m, n)    
    uy0 = zeros(m, n)    
    ρ0 = zeros(m, n)

    # Initializing the initial conditions
    for i = 1 : m, j = 1 : n
        ux0[i, j] = ux_funct(xs[i], ys[j])
        uy0[i, j] = uy_funct(xs[i], ys[j])
         ρ0[i, j] =  ρ_funct(xs[i], ys[j])
    end

    return xs, ys, Δx, Δy, ux0, uy0, ρ0 
end

"""
    Creates quasi-onedimensional initial conditions in x direction
"""
function quasi1dx(β, c, Lx, k::Int=1)
    function ux(x, y)
        return β * sin(k * x * 2 * π / Lx) + c
    end

    function uy(x, y)
        return 0.0
    end

    function ρ(x, y)
        return 1.0
    end

    return ux, uy, ρ
end

"""
    Creates quasi-onedimensional initial conditions in y direction
"""
function quasi1dy(β, c, Ly, k::Int=1)
    function ux(x, y)
        return 0.0
    end

    function uy(x, y)
        return β * sin(k * y * 2 * π / Ly) + c
    end

    function ρ(x, y)
        return 1.0
    end

    return ux, uy, ρ
end

# is zero for inputs smaller than 0 and goes to 1 
function cut_off(x, ϵ)
    function f(x) 
        if x > 0
            return exp(- 1 / x)
        else
            return 0.0
        end
    end
    return f(x / ϵ) / (f(x / ϵ) + f(1 - x / ϵ))
end


function two_sines_x(β1, c1, s1, β2, c2, s2, Lx, Ly, ϵ, uy0 = 0.0)
    # the weight of the first component
    function w1(y)
        return cut_off(y, ϵ) * cut_off(Ly / 2 -  y, ϵ)
    end
    function ux(x, y)
        return w1(y) * (β1 * sin((x - s1) * 2 * π / Lx) + c1) + (1 - w1(y)) * (β2 * sin((x - s2) * 2 * π / Lx) + c2)
    end

    function uy(x, y)
        return uy0
    end

    function ρ(x, y)
        return 1.0
    end

    return ux, uy, ρ
end

"""
    Computes the distance on the periodic domain [0, Lx] ×  [0, Ly]
"""
function pdist(x1, x2, y1, y2, Lx, Ly)
    return sqrt(min((x1 - x2)^2,  (x1 - x2 + Lx)^2, (x1 - x2 - Lx)^2 )  
               +min((y1 - y2)^2,  (y1 - y2 + Ly)^2, (y1 - y2 - Ly)^2 ))
end

""" 
    Creates a single Sedov-type blast wave with magnitude β, center (cx, cy), and standard deviations σ 
    ρ_atmos describes the density of the atmossphere
"""
function sedov(β, cx, cy, σ, Lx, Ly, ρ_atmos) 
    @assert 0 ≤ cx ≤ Lx
    @assert 0 ≤ cy ≤ Ly
    function ux(x, y)
        return 0.0
    end

    function uy(x, y)
        return 0.0
    end

    function ρ(x, y)
        return β / (2 * π) / σ^2 * exp(- pdist(x, cx, y, cy, Lx, Ly)^2 / 2 / σ^2) + ρ_atmos
    end

    return ux, uy, ρ
end

""" 
    Creates multiple Sedov-type blast wave with magnitudes βs, centers (cxs, cys), and standard deviations σs
    ρ_atmos describes the density of the atmossphere
"""
function sedov(βs::AbstractArray, cxs::AbstractArray, cys::AbstractArray, σs::AbstractArray, Lx, Ly, ρ_atmos) 
    @assert length(βs) == length(cxs) == length(cys) == length(σs) 
    function ux(x, y)
        return 0.0
    end

    function uy(x, y)
        return 0.0
    end

    fcts = [sedov(βs[i], cxs[i], cys[i], σs[i], Lx, Ly, 0.0) for i = 1 : length(βs)]

    function ρ(x, y)
        out = 0.0
        for i = 1 : length(βs)
            out += fcts[i][3](x, y)
        end
        return out + ρ_atmos
    end

    return ux, uy, ρ
end

"""
    Creates a swirl-like initial condition with veloxity proportional to β. The parameter γ determines the radius of
    the Gaussian cut-off
"""
function swirl(α, β, γα, γβ, ϵ)
    function ux(x, y)
        return (- α * (x - 0.5) * exp(- ((x - 0.5) ^ 2 + (y - 0.5) ^ 2) / 2 / γα^2) / sqrt(ϵ + norm([x - 0.5; y - 0.5])) + (β * (y - 0.5)) * exp(- ((x - 0.5) ^ 2 + (y - 0.5) ^ 2) / 2 / γβ^2)) / sqrt(ϵ + norm([x - 0.5; y - 0.5]))
    end

    function uy(x, y)
        return (- α * (y - 0.5) * exp(- ((x - 0.5) ^ 2 + (y - 0.5) ^ 2) / 2 / γα^2) / sqrt(ϵ + norm([x - 0.5; y - 0.5])) + (- β * (x - 0.5)) * exp(- ((x - 0.5) ^ 2 + (y - 0.5) ^ 2) / 2 / γβ^2) ) / sqrt(ϵ + norm([x - 0.5; y - 0.5]))
    end

    function ρ(x, y)
        return 1.0
    end

    return ux, uy, ρ
end


"""
    Creates a swirl-like initial condition with veloxity proportional to β. The parameter γ determines the radius of
    the Gaussian cut-off
"""
function seeded_swirl(α, β, γ, ϵ)
    Lx = 1.0
    Ly = 1.0
    function w1(y)
        return cut_off(y, ϵ) * cut_off(Ly / 2 -  y, ϵ)
    end
    function ux(x, y)
        return w1(y) * α
    end

    function uy(x, y)
        return 0.0
    end

    function ρ(x, y)
        return 1.0 - β * exp(- ((x - 0.5)^ 2 + (y - 0.5)^2) / 2 / γ^2)
    end

    return ux, uy, ρ
end
