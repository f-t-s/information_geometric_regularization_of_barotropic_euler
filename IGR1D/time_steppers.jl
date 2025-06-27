"""
    Implements the forward euler time stepper. Input variables are: sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    u0:        The initial x-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function forward_euler(sd, Δt, T, ρ0, u0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(u0)
    m = size(ρ0)[1]
    m_out = length(1 : subsamp : m)

    ρ  = copy(ρ0)
    μ = copy(u0) .* ρ0 

    k1_μ = zeros(m)
    k1_ρ  = zeros(m)
    
    # Setting up arrays for the output
    μ_out = Vector{stype}(copy(vec((u0 .* ρ0)[1 : subsamp : end, 1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end, 1 : subsamp : end]))
    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμ, ∂tρ, μ, ρ)
       ∂tρ .= 0.0; ∂tμ .= 0.0
       add_rhs!(∂tμ, ∂tρ, sd, μ, ρ)
    end

    while t < T 
        f!(k1_μ, k1_ρ, μ, ρ)
        @fastmath @inbounds @simd for i in eachindex(k1_μ)
            μ[i] += Δt * k1_μ[i]
            ρ[i] += Δt * k1_ρ[i] 
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μ_out, μ[1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μ_out ./ ρ_out, m_out, :), reshape(ρ_out, m_out, :), t_out
end

"""
    Implements the rk4 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    u0:        The initial x-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk4(sd, Δt, T, ρ0, u0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(u0)
    m = size(ρ0)[1]
    m_out = length(1 : subsamp : m)

    ρ = copy(ρ0)
    μ = copy(u0) .* ρ0 

    k1_μ = zeros(m)
    k1_ρ = zeros(m)
    k2_μ = zeros(m)
    k2_ρ = zeros(m)
    k3_μ = zeros(m)
    k3_ρ = zeros(m)
    k4_μ = zeros(m)
    k4_ρ = zeros(m)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μ = zeros(m)
    in2_ρ  = zeros(m)
    in3_μ = zeros(m)
    in3_ρ  = zeros(m)
    in4_μ = zeros(m)
    in4_ρ  = zeros(m)

    # Setting up arrays for the output
    μ_out = Vector{stype}(copy(vec((u0 .* ρ0)[1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end]))
    t_out = zeros(stype, 1) 

    t_out = zeros(stype, 1) 

    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμ, ∂tρ, μ, ρ)
       ∂tμ .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμ, ∂tρ, sd, μ, ρ)
    end

    while t < T 
        f!(k1_μ, k1_ρ, μ, ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in2_μ[i] = μ[i] + 0.5 * Δt * k1_μ[i]
            in2_ρ[i]  = ρ[i]  + 0.5 * Δt * k1_ρ[i]
        end

        f!(k2_μ, k2_ρ, in2_μ, in2_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in3_μ[i] = μ[i] + 0.5 * Δt * k2_μ[i]
            in3_ρ[i] = ρ[i]  + 0.5 * Δt * k2_ρ[i]
        end

        f!(k3_μ, k3_ρ, in3_μ, in3_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in4_μ[i] = μ[i] + 1.0 * Δt * k3_μ[i]
            in4_ρ[i] = ρ[i]  + 1.0 * Δt * k3_ρ[i]
        end

        f!(k4_μ, k4_ρ, in4_μ, in4_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            μ[i] +=  (Δt * 1/6 * k1_μ[i] 
                    + Δt * 1/3 * k2_μ[i] 
                    + Δt * 1/3 * k3_μ[i] 
                    + Δt * 1/6 * k4_μ[i])
            ρ[i]  += (Δt * 1/6 * k1_ρ[i] 
                    + Δt * 1/3 * k2_ρ[i] 
                    + Δt * 1/3 * k3_ρ[i] 
                    + Δt * 1/6 * k4_ρ[i])
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μ_out, μ[1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μ_out ./ ρ_out, m_out, :), reshape(ρ_out, m_out, :), t_out
end

"""
    Implements the rk3 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    u0:        The initial x-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk3(sd, Δt, T, ρ0, u0, uy0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(u0)
    m = size(ρ0)[1]
    m_out = length(1 : subsamp : m)

    ρ  = copy(ρ0)
    μ = copy(u0) .* ρ0 

    k1_μ = zeros(m)
    k1_ρ = zeros(m)
    k2_μ = zeros(m)
    k2_ρ = zeros(m)
    k3_μ = zeros(m)
    k3_ρ = zeros(m)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μ = zeros(m)
    in2_ρ = zeros(m)
    in3_μ = zeros(m)
    in3_ρ = zeros(m)

    # Setting up arrays for the output
    μ_out = Vector{stype}(copy(vec(u0 .* ρ0)))
    ρ_out = Vector{stype}(vec(ρ0))
    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμ, ∂tρ, μ, ρ)
       ∂tμ .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμ, ∂tρ, sd, μ, ρ)
    end

    while t < T 
        f!(k1_μ, k1_ρ, μ, ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in2_μ[i] = μ[i] + 0.5 * Δt * k1_μ[i]
            in2_ρ[i] = ρ[i] + 0.5 * Δt * k1_ρ[i]
        end
        f!(k2_μ, k2_ρ, in2_μ, in2_ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in3_μ[i] = μ[i] + 1.0 * Δt .* k2_μ[i] + 2.0 .* Δt .* k2_μ[i]
            in3_ρ[i] = ρ[i] + 1.0 * Δt .* k2_ρ[i] + 2.0 .* Δt .* k2_ρ[i]
        end
        f!(k3_μ, k3_ρ, in3_μ, in3_ρ)

        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            μ[i] += Δt * 1/6 * k1_μ[i] 
            μ[i] += Δt * 2/3 * k2_μ[i] 
            μ[i] += Δt * 1/6 * k3_μ[i] 
            ρ[i] += Δt * 1/6 * k1_ρ[i] 
            ρ[i] += Δt * 2/3 * k2_ρ[i] 
            ρ[i] += Δt * 1/6 * k3_ρ[i] 
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μ_out, μ[1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end])
            println("Current time snapshot is $(t)")
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μ_out ./ ρ_out, m_out, :), reshape(ρ_out, m_out, :), t_out
end

"""
    Implements the rk2 time stepper. Input variables are:
    sd:         The semidiscretization
    Δt:         The time step of the solver
    T:          The final time
    ρ0:         The initial density
    u0:        The initial x-velocity
    Δt_record:  The frequency with which intermediate results are saved
    verbose:    Turns on/off a print statemtent at each Δt_record
    stype:      The Floating Point data type used for storing the output
    subsamp:    Store only [1 : subsamp : end, 1 : subsamp : end] indices
"""
function rk2(sd, Δt, T, ρ0, u0, Δt_record, verbose=false, stype=Float64, subsamp=1)
    @assert size(ρ0) == size(u0)
    m = size(ρ0)[1]
    m_out = length(1 : subsamp : m)[1]

    ρ  = copy(ρ0)
    μ = copy(u0) .* ρ0 

    k1_μ = zeros(m)
    k1_ρ = zeros(m)
    k2_μ = zeros(m)
    k2_ρ = zeros(m)

    # setting up space for input to f!
    # in computing k2-k4 
    in2_μ = zeros(m)
    in2_ρ = zeros(m)

    # Setting up arrays for the output
    μ_out = Vector{stype}(copy(vec((u0 .* ρ0)[1 : subsamp : end])))
    ρ_out = Vector{stype}(vec(ρ0[1 : subsamp : end]))
    t_out = zeros(stype, 1) 


    # initialize the time
    t = Δt

    # miniature function for computing the time derivatives
    function f!(∂tμ, ∂tρ, μ, ρ)
       ∂tμ .= 0.0; ∂tρ .= 0.0
       add_rhs!(∂tμ, ∂tρ, sd, μ, ρ)
    end

    while t < T 
        f!(k1_μ, k1_ρ, μ, ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            in2_μ[i] = μ[i] + 2/3 * Δt * k1_μ[i]
            in2_ρ[i] = ρ[i] + 2/3 * Δt * k1_ρ[i]
        end
        f!(k2_μ, k2_ρ, in2_μ, in2_ρ)
        @inbounds @fastmath Threads.@threads for i in eachindex(μ)
            μ[i] += Δt * 1/4 * k1_μ[i] + Δt * 3/4 * k2_μ[i]
            ρ[i] += Δt * 1/4 * k1_ρ[i] + Δt * 3/4 * k2_ρ[i]
        end

        t += Δt
        # Write to output only if we have reached a new multiple of Δt_record
        if div(t, Δt_record) > div(t_out[end], Δt_record)   
            push!(t_out, t)
            append!(μ_out, μ[1 : subsamp : end])
            append!(ρ_out, ρ[1 : subsamp : end])
            if verbose
                println("Current time snapshot is $(t)")
            end
        end
    end
    return reshape(μ_out ./ ρ_out, m_out, :), reshape(ρ_out, m_out, :), t_out
end
