using DiffEqFlux, Flux

"""
    Module that modifies parameters 
"""
abstract type Modifier <: DiffEqFlux.FastLayer end

""" Required as a FastLayer, define a paramsum field """
paramlength(f::Modifier) = f.param_size

""" Required as a FastLayer """
initial_params(f::Modifier) = f.initial_params()

""" Nominal modifier does not change the parameters """
struct Nominal{P} <: Modifier
    name::String
    param_size::Int
    initial_params::P
    function Nominal(init = Flux.zeros)
        initial_params() = init(0)
        new{typeof(initial_params)}("Nominal", 0, initial_params)
    end
end

function (f::Nominal)(p_sub, p_link, p)
    return p_sub, p_link
end

function debug(f::Nominal, p_sub, p_link, p)
    return 0.0
end

include("iss_ctrnn0.jl")
include("iss_ctrnn1.jl")
