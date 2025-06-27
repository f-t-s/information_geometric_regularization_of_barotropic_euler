"""
    Initializes an m by n sized problem on an Lx domain
    domain, with initial conditions given by the functions
    u_funct, ρ_funct.
"""
function initialize(m, L, u_funct, ρ_funct) 
    # Compute grid spacing
    Δx = L / m
    
    # Compute the x and y coordinates we are adding a Δx / 2 and Δy / 2
    # so that the points are in the interior of [0, Lx] and [0, Ly]
    xs = Δx / 2 .+ (0 : (m - 1)) * Δx

    # Create arrays holding the initial conditions
    u0 = zeros(m)    
    ρ0 = zeros(m)

    # Initializing the initial conditions
    for i = 1 : m
        u0[i] = u_funct(xs[i])
        ρ0[i] = ρ_funct(xs[i])
    end

    return xs, Δx, u0, ρ0 
end

"""
    Creates sine wave initial condition
"""
function sine_wave(β, c, s, L, k::Int=1)
    function u(x)
        return β * sin(k * (x - s) * 2 * π / L) + c
    end

    function ρ(x)
        return 1.0
    end

    return u, ρ
end

"""
    Creates Riemann problem
"""
function riemann(β, c, s, k::Int=1)
    function u(x)
        return -β * tanh(k * (x - s)) + c
    end

    function ρ(x)
        return 1.0
    end

    return u, ρ
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

function shu_osher(L, ρ_l, ρ_r, u_l, u_r, β, k, ϵ)
    function u(x)
        return u_l * (cut_off(-(x - 0.2 * L), ϵ)) + u_r * (1 - cut_off(-(x - 0.2 * L), ϵ)) 
    end

    function ρ(x)
        return ρ_l * (cut_off(-(x - 0.2 * L), ϵ)) + (ρ_r + β *  sin(k * (x - 0.2 * L) * 2 * π / (0.8 * L) ) * cut_off(-(x - L), ϵ)) * (1 - cut_off(-(x - 0.2 * L), ϵ)) 
    end
    return u, ρ
end

"""
    Creates an array of length m that interpolates between vl and vr, using the tanh 
"""
function interpol(vl, vr, m, dst = 3)
    xs = range(-dst, dst, m)
    return (vr + vl) / 2 .+ tanh.(xs) * (vr - vl) / (tanh(xs[end]) - tanh(xs[1]))
end

"""
    Initializes an m by n sized problem on an Lx domain
    domain, with initial conditions given by the functions
    u_funct, ρ_funct.
"""
function initialize(m, L, u_funct, ρ_funct, padding::Int, dst=3.0) 
    # Compute grid spacing
    Δx = L / m
    
    # Compute the x and y coordinates we are adding a Δx / 2 and Δy / 2
    # so that the points are in the interior of [0, Lx] and [0, Ly]
    xs = Δx / 2 .+ (0 : (m - 1)) * Δx

    # Create arrays holding the initial conditions
    u0 = zeros(m)    
    ρ0 = zeros(m)

    # Initializing the initial conditions
    for i = 1 : m
        u0[i] = u_funct(xs[i])
        ρ0[i] = ρ_funct(xs[i])
    end

    xs_pad_left = xs[1] .- ((padding * m) : -1 : 1) * Δx
    xs_pad_right = xs[end] .+ (1 : ((padding + 1) * m)) * Δx
    xs = vcat(xs_pad_left, xs, xs_pad_right)

    u0_pad_left = fill(u0[1], padding * m)
    u0_pad_right = fill(u0[end], padding * m)
    u0_interpol = interpol(u0[end], u0[1], m, dst)
    u0 = vcat(u0_pad_left, u0, u0_pad_right, u0_interpol)
    
    ρ0_pad_left = fill(ρ0[1], padding * m)
    ρ0_pad_right = fill(ρ0[end], padding * m)
    ρ0_interpol = interpol(ρ0[end], ρ0[1], m, dst)
    ρ0 = vcat(ρ0_pad_left, ρ0, ρ0_pad_right, ρ0_interpol)

    return xs, Δx, u0, ρ0, (padding * m + 1) : (padding * m + m) 
end
