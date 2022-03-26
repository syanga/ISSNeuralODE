using DiffEqFlux, Flux
""" CTRNN1 implements a continuous time recurrent neural net
    with 1 hidden layer:
        dxdt = -x/τ + Wσ(Ax + Bu + μ) + ν

    Available types:
        * CTRNN1Nominal             dxdt = -x/τ + Wσ(Ax + Bu + μ) + ν
        * CTRNN1Canonical           dxdt = -x/τ + W(σ(Ax + Bu + μ) - σ(μ))
        * CTRNN1Zero                dxdt = -x/τ + Wσ(Ax + Bu)
"""
abstract type CTRNN1 <: Ode end


""" dxdt = -x/τ + Wσ(Ax + Bu + μ) + ν 
    CTRNN1Nominal """
struct CTRNN1Nominal{F,FU,F2} <: CTRNN1
    state_dim::Int
    hidden_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN1Nominal(
        state_dim::Int,
        hidden_dim::Int,
        input_dim::Int;
        σ = relu,
        initW = Flux.glorot_uniform,
        initb = Flux.zeros,
        σ_u = identity,
    )
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["logτ"] = i:i
        i += 1
        index_dict["W"] = i:i+state_dim*hidden_dim-1
        i += state_dim * hidden_dim
        index_dict["A"] = i:i+hidden_dim*state_dim-1
        i += hidden_dim * state_dim
        index_dict["B"] = i:i+hidden_dim*input_dim-1
        i += hidden_dim * input_dim
        index_dict["μ"] = i:i+hidden_dim-1
        i += hidden_dim
        index_dict["ν"] = i:i+state_dim-1
        i += state_dim

        # initialize initial parameters
        initial_params() = vcat(
            Flux.zeros(1),
            vec(initW(state_dim, hidden_dim)),
            vec(initW(hidden_dim, state_dim)),
            vec(initW(hidden_dim, input_dim)),
            vec(initb(hidden_dim)),
            vec(initb(state_dim)),
        )

        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            hidden_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN1Nominal, p)
    logτ = view(p, f.index_dict["logτ"])
    W = reshape(view(p, f.index_dict["W"]), f.state_dim, f.hidden_dim)
    A = reshape(view(p, f.index_dict["A"]), f.hidden_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.hidden_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    ν = view(p, f.index_dict["ν"])
    return logτ, W, A, B, μ, ν
end

function (f::CTRNN1Nominal)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ, W, A, B, μ, ν = unpack_params(f, p)
    return -x .* exp.(-logτ) .+ (W * f.σ.((A * x .+ B * f.σ_u.(u)) .+ μ) .+ ν)
end


""" dxdt = -x/τ + W(σ(Ax + Bu + μ) - σ(μ))
    CTRNN1Canonical """
struct CTRNN1Canonical{F,FU,F2} <: CTRNN1
    state_dim::Int
    hidden_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN1Canonical(
        state_dim::Int,
        hidden_dim::Int,
        input_dim::Int;
        σ = relu,
        initW = Flux.glorot_uniform,
        initb = Flux.zeros,
        σ_u = identity,
    )
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["logτ"] = i:i
        i += 1
        index_dict["W"] = i:i+state_dim*hidden_dim-1
        i += state_dim * hidden_dim
        index_dict["A"] = i:i+hidden_dim*state_dim-1
        i += hidden_dim * state_dim
        index_dict["B"] = i:i+hidden_dim*input_dim-1
        i += hidden_dim * input_dim
        index_dict["μ"] = i:i+hidden_dim-1
        i += hidden_dim

        # initialize initial parameters
        function initial_params()
            W = initW(state_dim, hidden_dim)
            A = initW(hidden_dim, state_dim)
            return vcat(
                Flux.zeros(1),
                vec(W),
                vec(A / maximum(0.5 * eigen(A * W + W' * A').values)),
                vec(initW(hidden_dim, input_dim)),
                vec(initb(hidden_dim)),
            )
        end
        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            hidden_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN1Canonical, p)
    logτ = view(p, f.index_dict["logτ"])
    W = reshape(view(p, f.index_dict["W"]), f.state_dim, f.hidden_dim)
    A = reshape(view(p, f.index_dict["A"]), f.hidden_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.hidden_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    ν = -W * f.σ.(μ)
    return logτ, W, A, B, μ, ν
end

function (f::CTRNN1Canonical)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ = view(p, f.index_dict["logτ"])
    W = reshape(view(p, f.index_dict["W"]), f.state_dim, f.hidden_dim)
    A = reshape(view(p, f.index_dict["A"]), f.hidden_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.hidden_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    return -x .* exp.(-logτ) .+ W * (f.σ.(A * x .+ B * f.σ_u.(u) .+ μ) .- f.σ.(μ))
end


""" dxdt = -x/τ + W(σ(Ax + Bu + μ) - σ(μ))
    CTRNN1Zero """
struct CTRNN1Zero{F,FU,F2} <: CTRNN1
    state_dim::Int
    hidden_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN1Zero(
        state_dim::Int,
        hidden_dim::Int,
        input_dim::Int;
        σ = tanh,
        initW = Flux.glorot_uniform,
        initb = Flux.zeros,
        σ_u = identity,
    )
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["logτ"] = i:i
        i += 1
        index_dict["W"] = i:i+state_dim*hidden_dim-1
        i += state_dim * hidden_dim
        index_dict["A"] = i:i+hidden_dim*state_dim-1
        i += hidden_dim * state_dim
        index_dict["B"] = i:i+hidden_dim*input_dim-1
        i += hidden_dim * input_dim

        # initialize initial parameters
        initial_params() = vcat(
            Flux.zeros(1),
            vec(initW(state_dim, hidden_dim)),
            vec(initW(hidden_dim, state_dim)),
            vec(initW(hidden_dim, input_dim)),
        )

        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            hidden_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN1Zero, p)
    logτ = view(p, f.index_dict["logτ"])
    W = reshape(view(p, f.index_dict["W"]), f.state_dim, f.hidden_dim)
    A = reshape(view(p, f.index_dict["A"]), f.hidden_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.hidden_dim, f.input_dim)
    μ = zeros(f.hidden_dim)
    ν = zeros(f.state_dim)
    return logτ, W, A, B, μ, ν
end

function (f::CTRNN1Zero)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ = view(p, f.index_dict["logτ"])
    W = reshape(view(p, f.index_dict["W"]), f.state_dim, f.hidden_dim)
    A = reshape(view(p, f.index_dict["A"]), f.hidden_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.hidden_dim, f.input_dim)
    return -x .* exp.(-logτ) .+ W * f.σ.(A * x .+ B * f.σ_u.(u))
end
