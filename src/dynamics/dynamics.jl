abstract type Ode <: DiffEqFlux.FastLayer end

include("ctrnn0.jl")
include("ctrnn1.jl")
include("affine.jl")

""" Required as a FastLayer, define a paramsum field """
paramlength(f::Ode) = f.paramsum

""" Required as a FastLayer """
initial_params(f::Ode) = f.initial_params()

""" convert xu concatenation to x,u in 1d case """
@inline function extract_inputs(f::O, xu::AbstractArray{<:T,1}) where {T} where {O<:Ode}
    x = view(xu, 1:f.state_dim)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim)
    return x, u
end

""" convert xu concatenation to x,u in 2d case """
@inline function extract_inputs(f::O, xu::AbstractArray{<:T,2}) where {T} where {O<:Ode}
    x = view(xu, 1:f.state_dim, :)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim, :)
    return x, u
end

""" convert xu concatenation to x,u in 3d case """
@inline function extract_inputs(f::O, xu::AbstractArray{<:T,3}) where {T} where {O<:Ode}
    x = view(xu, 1:f.state_dim, :, :)
    u = view(xu, f.state_dim+1:f.state_dim+f.input_dim, :, :)
    return x, u
end
