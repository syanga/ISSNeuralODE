using DiffEqFlux, Flux
abstract type AffineSystem <: Ode end


""" dxdt = Ax + Bu + ν
    AffineOde """
struct AffineOde{FU,F2} <: AffineSystem
    state_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String, UnitRange{Int}}
    σ_u::FU    
    initial_params::F2
    function AffineOde(state_dim::Integer, input_dim::Integer; initW=Flux.glorot_uniform, initb=Flux.zeros, σ_u=identity)
        # compute indices
        i = 1
        index_dict = Dict{String, UnitRange{Int}}()
        index_dict["A"] = i:i+state_dim*state_dim-1
        i += state_dim*state_dim
        index_dict["B"] = i:i+input_dim*state_dim-1
        i += input_dim*state_dim
        index_dict["ν"] = i:i+state_dim-1
        i += state_dim

        # initialize initial parameters
        initial_params() = vcat(
            vec(initW(state_dim, state_dim)), 
            vec(initW(state_dim, input_dim)),
            vec(initb(state_dim)))

        new{typeof(σ_u),typeof(initial_params)}(state_dim, input_dim, i-1, index_dict, σ_u, initial_params)
    end
end

function unpack_params(f::AffineOde, p)
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    ν = view(p, f.index_dict["ν"])
    return A, B, ν
end

function (f::AffineOde)(xu, p)
    x,u = extract_inputs(f,xu)
    A, B, ν = unpack_params(f, p)
    return A*x .+ B*f.σ_u.(u) .+ ν
end


""" dxdt = Ax + Bu
    LinearOde """
struct LinearOde{FU,F2} <: AffineSystem
    state_dim::Int
    input_dim::Int
    paramsum::Int
    index_dict::Dict{String, UnitRange{Int}}
    σ_u::FU
    initial_params::F2
    function LinearOde(state_dim::Integer, input_dim::Integer; initW=Flux.glorot_uniform, σ_u=identity)
        # compute indices
        i = 1
        index_dict = Dict{String, UnitRange{Int}}()
        index_dict["A"] = i:i+state_dim*state_dim-1
        i += state_dim*state_dim
        index_dict["B"] = i:i+input_dim*state_dim-1
        i += input_dim*state_dim

        # initialize initial parameters
        initial_params() = vcat(
            vec(initW(state_dim, state_dim)), 
            vec(initW(state_dim, input_dim)))

        new{typeof(σ_u),typeof(initial_params)}(state_dim, input_dim, i-1, index_dict, σ_u, initial_params)
    end
end

function unpack_params(f::LinearOde, p)
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    ν = zeros(f.state_dim)
    return A, B, ν
end

function (f::LinearOde)(xu, p)
    x,u = extract_inputs(f,xu)
    A = reshape(view(p, f.index_dict["A"]), f.state_dim, f.state_dim)
    B = reshape(view(p, f.index_dict["B"]), f.state_dim, f.input_dim)
    return A*x .+ B*f.σ_u.(u)
end
