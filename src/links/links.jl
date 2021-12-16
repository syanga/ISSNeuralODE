# module links
# """
#     Links: state,input -> output for each subsystem, of type DiffEqFlux.FastLayer
    
#     interface includes function of form h(xu, p) where xu is a concatenation of state and input, parameters p
# """
# using DiffEqFlux

# abstract type Link <: DiffEqFlux.FastLayer end

# include("affine_canonical.jl")
# export AffineFeedthrough, LinearFeedthrough, AffineCanonical, LinearCanonical, AffineResistive, LinearResistive
# export paramlength, initial_params, convert_verilogA

# end # module

using DiffEqFlux

abstract type Link <: DiffEqFlux.FastLayer end

include("affine_layer.jl")

""" Required as a FastLayer, define a paramsum field """
paramlength(f::AffineLayer) = f.paramsum

""" Required as a FastLayer """
initial_params(f::AffineLayer) = f.initial_params()

""" convert xu concatenation to x,u in 1d case """
@inline function extract_inputs(f::L, xu::AbstractArray{T,1}) where T where L <: Link
    x = view(xu, 1:f.state_dim)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim)
    return x,u
end

""" convert xu concatenation to x,u in 2d case """
@inline function extract_inputs(f::L, xu::AbstractArray{<:T,2}) where T where L <: Link
    x = view(xu, 1:f.state_dim,:)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim,:)
    return x,u
end

""" convert xu concatenation to x,u in 3d case """
@inline function extract_inputs(f::L, xu::AbstractArray{T,3}) where T where L <: Link
    x = view(xu, 1:f.state_dim,:,:)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim,:,:)
    return x,u
end

""" Identity Link"""
struct IdentityLink{I} <: Link
    initial_params::I
    paramsum::Int
    function IdentityLink(initb=Flux.zeros)
        initial_params() = initb(0)
        new{typeof(initial_params)}(initial_params, 0)
    end
end

function (f::IdentityLink)(x, p)
    return x
end
