using LinearAlgebra

""" AffineLayer implements the mapping
        y = Cx + Du + b
    using the function y = AffineLayer(xu,p), where 
    xu is the concatenation of state x and input u and
    p are the parameters. Should also have a function 
    unpack_params(f::AffineLayer, p) that converts 
    parameter vector p into relevant parameter matrices

    Available types:
        * AffineCanonical                       y = Cx + b
        * LinearCanonical                       y = Cx
"""
abstract type AffineLayer <: Link end


""" y = Cx + b
    AffineCanonical """
struct AffineCanonical{F,S} <: AffineLayer
    state_dim::Int
    output_dim::Int
    paramsum::Int
    index_dict::Dict{String, UnitRange{Int}}
    initial_params::F
    σ::S
    function AffineCanonical(state_dim::Int, output_dim::Int;
                             initW=Flux.glorot_uniform, initb=Flux.zeros, σ=identity)
    # compute indices
    i = 1
    index_dict = Dict{String, UnitRange{Int}}()
    index_dict["C"] = i:i+output_dim*state_dim-1
    i += output_dim*state_dim
    index_dict["b"] = i:i+output_dim-1
    i += output_dim

    # initialize initial parameters
    initial_params() = vcat(
        vec(initW(output_dim, state_dim)), 
        initb(output_dim)
    )

    new{typeof(initial_params), typeof(σ)}(state_dim, output_dim, i-1, index_dict, initial_params, σ)
    end
end

function unpack_params(f::AffineCanonical, p)
    C = reshape(p[f.index_dict["C"]], f.output_dim, f.state_dim)
    D = zeros(f.output_dim, f.output_dim)
    b = p[f.index_dict["b"]]
    return C,D,b
end

function (f::AffineCanonical)(x, p)
    C = @view p[reshape(f.index_dict["C"], f.output_dim, f.state_dim)]
    b = @view p[f.index_dict["b"]]
    return C*f.σ.(x) .+ b
end

""" y = Cx
    LinearCanonical """
struct LinearCanonical{F} <: AffineLayer
    state_dim::Int
    output_dim::Int
    paramsum::Int
    index_dict::Dict{String, UnitRange{Int}}
    initial_params::F
    function LinearCanonical(state_dim::Int, output_dim::Int; initW=Flux.glorot_uniform)
        # compute indices
        i = 1
        index_dict = Dict{String, UnitRange{Int}}()
        index_dict["C"] = i:i+output_dim*state_dim-1
        i += output_dim*state_dim

        # initialize initial parameters
        initial_params() = vec(initW(output_dim, state_dim))

        new{typeof(initial_params)}(state_dim, output_dim, i-1, index_dict, initial_params)
    end
end

function unpack_params(f::LinearCanonical, p)
    C = reshape(p[f.index_dict["C"]], f.output_dim, f.state_dim)
    D = zeros(f.output_dim, f.output_dim)
    b = zeros(f.output_dim)
    return C,D,b
end

function (f::LinearCanonical)(x, p)
    C = p[reshape(f.index_dict["C"], f.output_dim, f.state_dim)]
    return C*x
end
