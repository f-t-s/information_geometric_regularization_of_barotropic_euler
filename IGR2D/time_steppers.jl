"""
    Implements the forward euler time stepper. Input variables are: sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    ux0:        The initial x-velocity
    uy0:        The initial y-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function forward_euler(sd, Δt, T, ρ0, ux0, uy0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(ux0) == size(uy0)
    m, n = size(ρ0)
    m_out = length(1 : subsamp : m)
    n_out = length(1 : subsamp : n)

    ρ  = copy(ρ0)
    μx = copy(ux0) .* ρ0 
    μy = copy(uy0) .* ρ0 

    k1_μx = zeros(m, n)
    k1_μy = zeros(m, n)
    k1_ρ  = zeros(m, n)
    
    # Setting up arrays for the output
    μx_out = Vector{stype}(copy(vec((ux0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    μy_out = Vector{stype}(copy(vec((uy0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end, 1 : subsamp : end]))
    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμx, ∂tμy, ∂tρ, μx, μy, ρ)
       ∂tρ .= 0.0; ∂tμx .= 0.0; ∂tμy .= 0.0 
       add_rhs!(∂tμx, ∂tμy, ∂tρ, sd, μx, μy, ρ)
    end

    while t < T 
        f!(k1_μx, k1_μy, k1_ρ, μx, μy, ρ)
        @fastmath @inbounds @simd for i in eachindex(k1_μx)
            μx[i] += Δt * k1_μx[i]
            μy[i] += Δt * k1_μy[i] 
            ρ[i]  += Δt * k1_ρ[i] 
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μx_out, μx[1 : subsamp : end, 1 : subsamp : end])
            append!(μy_out, μy[1 : subsamp : end, 1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end, 1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μx_out ./ ρ_out, m_out, n_out, :), reshape(μy_out ./ ρ_out, m_out, n_out, :), reshape(ρ_out, m_out, n_out, :), t_out
end

"""
    Implements the rk4 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    ux0:        The initial x-velocity
    uy0:        The initial y-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk4(sd, Δt, T, ρ0, ux0, uy0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(ux0) == size(uy0)
    m, n = size(ρ0)
    m_out = length(1 : subsamp : m)
    n_out = length(1 : subsamp : n)

    ρ  = copy(ρ0)
    μx = copy(ux0) .* ρ0 
    μy = copy(uy0) .* ρ0 

    k1_μx = zeros(m, n)
    k1_μy = zeros(m, n)
    k1_ρ  = zeros(m, n)
    k2_μx = zeros(m, n)
    k2_μy = zeros(m, n)
    k2_ρ  = zeros(m, n)
    k3_μx = zeros(m, n)
    k3_μy = zeros(m, n)
    k3_ρ  = zeros(m, n)
    k4_μx = zeros(m, n)
    k4_μy = zeros(m, n)
    k4_ρ  = zeros(m, n)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μx = zeros(m, n)
    in2_μy = zeros(m, n)
    in2_ρ  = zeros(m, n)
    in3_μx = zeros(m, n)
    in3_μy = zeros(m, n)
    in3_ρ  = zeros(m, n)
    in4_μx = zeros(m, n)
    in4_μy = zeros(m, n)
    in4_ρ  = zeros(m, n)

    # Setting up arrays for the output
    μx_out = Vector{stype}(copy(vec((ux0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    μy_out = Vector{stype}(copy(vec((uy0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end, 1 : subsamp : end]))
    t_out = zeros(stype, 1) 


    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμx, ∂tμy, ∂tρ, μx, μy, ρ)
       ∂tμx .= 0.0; ∂tμy .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμx, ∂tμy, ∂tρ, sd, μx, μy, ρ)
    end

    while t < T 
        f!(k1_μx, k1_μy, k1_ρ, μx,                    μy, ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in2_μx[i] = μx[i] + 0.5 * Δt * k1_μx[i]
            in2_μy[i] = μy[i] + 0.5 * Δt * k1_μy[i]
            in2_ρ[i]  = ρ[i]  + 0.5 * Δt * k1_ρ[i]
        end

        f!(k2_μx, k2_μy, k2_ρ, in2_μx, in2_μy, in2_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in3_μx[i] = μx[i] + 0.5 * Δt * k2_μx[i]
            in3_μy[i] = μy[i] + 0.5 * Δt * k2_μy[i]
            in3_ρ[i]  = ρ[i]  + 0.5 * Δt * k2_ρ[i]
        end

        f!(k3_μx, k3_μy, k3_ρ, in3_μx, in3_μy, in3_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in4_μx[i] = μx[i] + 1.0 * Δt * k3_μx[i]
            in4_μy[i] = μy[i] + 1.0 * Δt * k3_μy[i]
            in4_ρ[i]  = ρ[i]  + 1.0 * Δt * k3_ρ[i]
        end

        f!(k4_μx, k4_μy, k4_ρ, in4_μx, in4_μy, in4_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            μx[i] += (Δt * 1/6 * k1_μx[i] 
                    + Δt * 1/3 * k2_μx[i] 
                    + Δt * 1/3 * k3_μx[i] 
                    + Δt * 1/6 * k4_μx[i])
            μy[i] += (Δt * 1/6 * k1_μy[i] 
                    + Δt * 1/3 * k2_μy[i] 
                    + Δt * 1/3 * k3_μy[i] 
                    + Δt * 1/6 * k4_μy[i])
            ρ[i]  += (Δt * 1/6 * k1_ρ[i] 
                    + Δt * 1/3 * k2_ρ[i] 
                    + Δt * 1/3 * k3_ρ[i] 
                    + Δt * 1/6 * k4_ρ[i])
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μx_out, μx[1 : subsamp : end, 1 : subsamp : end])
            append!(μy_out, μy[1 : subsamp : end, 1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end, 1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μx_out ./ ρ_out, m_out, n_out, :), reshape(μy_out ./ ρ_out, m_out, n_out, :), reshape(ρ_out, m_out, n_out, :), t_out
end

"""
    Implements the rk3 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    ux0:        The initial x-velocity
    uy0:        The initial y-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk3(sd, Δt, T, ρ0, ux0, uy0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(ux0) == size(uy0)
    m, n = size(ρ0)
    m_out = length(1 : subsamp : m)
    n_out = length(1 : subsamp : n)

    ρ  = copy(ρ0)
    μx = copy(ux0) .* ρ0 
    μy = copy(uy0) .* ρ0 

    k1_μx = zeros(m, n)
    k1_μy = zeros(m, n)
    k1_ρ  = zeros(m, n)
    k2_μx = zeros(m, n)
    k2_μy = zeros(m, n)
    k2_ρ  = zeros(m, n)
    k3_μx = zeros(m, n)
    k3_μy = zeros(m, n)
    k3_ρ  = zeros(m, n)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μx = zeros(m, n)
    in2_μy = zeros(m, n)
    in2_ρ  = zeros(m, n)
    in3_μx = zeros(m, n)
    in3_μy = zeros(m, n)
    in3_ρ  = zeros(m, n)

    # Setting up arrays for the output
    μx_out = Vector{stype}(copy(vec(ux0 .* ρ0)))
    μy_out = Vector{stype}(copy(vec(uy0 .* ρ0)))
    ρ_out = Vector{stype}(vec(ρ0))
    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμx, ∂tμy, ∂tρ, μx, μy, ρ)
       ∂tμx .= 0.0; ∂tμy .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμx, ∂tμy, ∂tρ, sd, μx, μy, ρ)
    end

    while t < T 
        f!(k1_μx, k1_μy, k1_ρ, μx,                    μy, ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in2_μx[i] = μx[i] + 0.5 * Δt * k1_μx[i]
            in2_μy[i] = μy[i] + 0.5 * Δt * k1_μy[i]
            in2_ρ[i]  = ρ[i]  + 0.5 * Δt * k1_ρ[i]
        end
        f!(k2_μx, k2_μy, k2_ρ, in2_μx, in2_μy, in2_ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in3_μx[i] = μx[i] + 1.0 * Δt .* k2_μx[i] + 2.0 .* Δt .* k2_μx[i]
            in3_μy[i] = μy[i] + 1.0 * Δt .* k2_μy[i] + 2.0 .* Δt .* k2_μy[i]
            in3_ρ[i]  = ρ[i]  + 1.0 * Δt .* k2_ρ[i]  + 2.0 .* Δt .* k2_ρ[i]
        end
        f!(k3_μx, k3_μy, k3_ρ, in3_μx, in3_μy, in3_ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            μx[i] += Δt * 1/6 * k1_μx[i] 
            μx[i] += Δt * 2/3 * k2_μx[i] 
            μx[i] += Δt * 1/6 * k3_μx[i] 
            μy[i] += Δt * 1/6 * k1_μy[i] 
            μy[i] += Δt * 2/3 * k2_μy[i] 
            μy[i] += Δt * 1/6 * k3_μy[i] 
            ρ[i]  += Δt * 1/6 * k1_ρ[i] 
            ρ[i]  += Δt * 2/3 * k2_ρ[i] 
            ρ[i]  += Δt * 1/6 * k3_ρ[i] 
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μx_out, μx[1 : subsamp : end, 1 : subsamp : end])
            append!(μy_out, μy[1 : subsamp : end, 1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end, 1 : subsamp : end])
            println("Current time snapshot is $(t)")
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μx_out ./ ρ_out, m_out, n_out, :), reshape(μy_out ./ ρ_out, m_out, n_out, :), reshape(ρ_out, m_out, n_out, :), t_out
end

"""
    Implements the rk2 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    ux0:        The initial x-velocity
    uy0:        The initial y-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk2(sd, Δt, T, ρ0, ux0, uy0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(ux0) == size(uy0)
    m, n = size(ρ0)
    m_out = length(1 : subsamp : m)
    n_out = length(1 : subsamp : n)

    ρ  = copy(ρ0)
    μx = copy(ux0) .* ρ0 
    μy = copy(uy0) .* ρ0 

    k1_μx = zeros(m, n)
    k1_μy = zeros(m, n)
    k1_ρ  = zeros(m, n)
    k2_μx = zeros(m, n)
    k2_μy = zeros(m, n)
    k2_ρ  = zeros(m, n)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μx = zeros(m, n)
    in2_μy = zeros(m, n)
    in2_ρ  = zeros(m, n)

    # Setting up arrays for the output
    μx_out = Vector{stype}(copy(vec((ux0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    μy_out = Vector{stype}(copy(vec((uy0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end, 1 : subsamp : end]))
    t_out = zeros(stype, 1) 




    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμx, ∂tμy, ∂tρ, μx, μy, ρ)
       ∂tμx .= 0.0; ∂tμy .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμx, ∂tμy, ∂tρ, sd, μx, μy, ρ)
    end

    while t < T 
        f!(k1_μx, k1_μy, k1_ρ, μx,                    μy, ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            in2_μx[i] = μx[i] + 2/3 * Δt * k1_μx[i]
            in2_μy[i] = μy[i] + 2/3 * Δt * k1_μy[i]
            in2_ρ[i]  = ρ[i]  + 2/3 * Δt * k1_ρ[i]
        end
        f!(k2_μx, k2_μy, k2_ρ, in2_μx, in2_μy, in2_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μx)
            μx[i] += Δt * 1/4 * k1_μx[i] + Δt * 3/4 * k2_μx[i]
            μy[i] += Δt * 1/4 * k1_μy[i] + Δt * 3/4 * k2_μy[i]
             ρ[i] += Δt * 1/4 * k1_ρ[i] + Δt * 3/4 * k2_ρ[i]
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μx_out, μx[1 : subsamp : end, 1 : subsamp : end])
            append!(μy_out, μy[1 : subsamp : end, 1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end, 1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μx_out ./ ρ_out, m_out, n_out, :), reshape(μy_out ./ ρ_out, m_out, n_out, :), reshape(ρ_out, m_out, n_out, :), t_out
end

