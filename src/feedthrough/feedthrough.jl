using DiffEqFlux

abstract type Feedthrough <: DiffEqFlux.FastLayer end

include("linear_feedthrough.jl")

""" Required as a FastLayer, define a paramsum field """
paramlength(f::Feedthrough) = f.paramsum

""" Required as a FastLayer """
initial_params(f::Feedthrough) = f.initial_params()


""" Identity Link"""
struct NoFeedthrough{I} <: Feedthrough
    initial_params::I
    paramsum::Int
    function NoFeedthrough(initb=Flux.zeros)
        initial_params() = initb(0)
        new{typeof(initial_params)}(initial_params, 0)
    end
end

function unpack_params(f::NoFeedthrough, p)
    return false
end

function (f::NoFeedthrough)(ȳ, u, p)
    return ȳ
end
