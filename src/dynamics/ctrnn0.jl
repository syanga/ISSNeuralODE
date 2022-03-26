using DiffEqFlux, Flux
""" CTRNN0 implements a continuous time recurrent neural net
    with no hidden layers:
        dxdt = -x/τ + σ(Ax + Bu + μ) + ν

    Available types:
        * CTRNN0Elman               dxdt = -x/τ + σ(Ax + Bu + μ) + ν
        * CTRNN0Canonical           dxdt = -x/τ + σ(Ax + Bu + μ) - σ(μ)
        * CTRNN0Zero                dxdt = -x/τ + σ(Ax + Bu)
"""
abstract type CTRNN0 <: Ode end


""" dxdt = -x/τ + σ(Ax + Bu + μ) + ν
    CTRNN0Elman """
struct CTRNN0Elman{F,FU,F2} <: CTRNN0
    state_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN0Elman(
        state_dim::Integer,
        input_dim::Integer;
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
        index_dict["A"] = i:i+state_dim*state_dim-1
        i += state_dim * state_dim
        index_dict["B"] = i:i+input_dim*state_dim-1
        i += input_dim * state_dim
        index_dict["μ"] = i:i+state_dim-1
        i += state_dim
        index_dict["ν"] = i:i+state_dim-1
        i += state_dim

        # initialize initial parameters
        initial_params() = vcat(
            Flux.zeros(1),
            vec(initW(state_dim, state_dim)),
            vec(initW(state_dim, input_dim)),
            vec(initb(state_dim)),
            vec(initb(state_dim)),
        )

        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN0Elman, p)
    logτ = view(p, f.index_dict["logτ"])
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    ν = view(p, f.index_dict["ν"])
    return logτ, A, B, μ, ν
end

function (f::CTRNN0Elman)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ, A, B, μ, ν = unpack_params(f, p)
    return -x .* exp.(-logτ) .+ (f.σ.((A * x .+ B * f.σ_u.(u)) .+ μ) .+ ν)
end


""" dxdt = -x/τ + σ(Ax + Bu + μ) - σ(μ)
    CTRNN0Canonical """
struct CTRNN0Canonical{F,FU,F2} <: CTRNN0
    state_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN0Canonical(
        state_dim::Integer,
        input_dim::Integer;
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
        index_dict["A"] = i:i+state_dim*state_dim-1
        i += state_dim * state_dim
        index_dict["B"] = i:i+input_dim*state_dim-1
        i += input_dim * state_dim
        index_dict["μ"] = i:i+state_dim-1
        i += state_dim

        # initialize initial parameters
        initial_params() = vcat(
            Flux.zeros(1),
            vec(initW(state_dim, state_dim)),
            vec(initW(state_dim, input_dim)),
            vec(initb(state_dim)),
        )

        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN0Canonical, p)
    logτ = view(p, f.index_dict["logτ"])
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    ν = -f.σ.(μ)
    return logτ, A, B, μ, ν
end

function (f::CTRNN0Canonical)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ = view(p, f.index_dict["logτ"])
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    μ = view(p, f.index_dict["μ"])
    # return -x.*exp.(-logτ) .+ f.σ.(A*x .+ B*f.σ_u.(u) .+ μ) .- f.σ.(μ)
    return -(x .* exp.(-logτ) .+ f.σ.(μ)) .+ f.σ.((A * x .+ B * f.σ_u.(u)) .+ μ)
end


""" dxdt = -x/τ + σ(Ax + Bu)
    CTRNN0Zero """
struct CTRNN0Zero{F,FU,F2} <: CTRNN0
    state_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    σ::F
    σ_u::FU
    initial_params::F2
    function CTRNN0Zero(
        state_dim::Integer,
        input_dim::Integer;
        σ = tanh,
        initW = Flux.glorot_uniform,
        σ_u = identity,
    )
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["logτ"] = i:i
        i += 1
        index_dict["A"] = i:i+state_dim*state_dim-1
        i += state_dim * state_dim
        index_dict["B"] = i:i+input_dim*state_dim-1
        i += input_dim * state_dim

        # initialize initial parameters
        initial_params() = vcat(
            Flux.zeros(1),
            vec(initW(state_dim, state_dim)),
            vec(initW(state_dim, input_dim)),
        )

        new{typeof(σ),typeof(σ_u),typeof(initial_params)}(
            state_dim,
            input_dim,
            i - 1,
            index_dict,
            σ,
            σ_u,
            initial_params,
        )
    end
end

function unpack_params(f::CTRNN0Zero, p)
    logτ = view(p, f.index_dict["logτ"])
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    μ = zeros(f.state_dim)
    ν = zeros(f.state_dim)
    return logτ, A, B, μ, ν
end

function (f::CTRNN0Zero)(xu, p)
    x, u = extract_inputs(f, xu)
    logτ = view(p, f.index_dict["logτ"])
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    return -x .* exp.(-logτ) .+ f.σ.(A * x .+ B * f.σ_u.(u))
end
