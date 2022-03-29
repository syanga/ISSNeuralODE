abstract type Interpolator end


"""
    Batch a bunch of nonuniform interpolators
"""
struct BatchedInterpolator{F,T} <: Interpolator
    batch_size::Int
    input_dim::Int
    interpolators::F
    t_span::Tuple{T,T}
    function BatchedInterpolator(interpolators)
        batch_size = length(interpolators)

        # assume all interpolators match first, use to set bounds and datatypes
        t_span = interpolators[1].t_span
        new{typeof(Tuple(interpolators)),typeof(t_span[1])}(
            batch_size,
            input_dim,
            Tuple(interpolators),
            t_span,
        )
    end
end

function (f::BatchedInterpolator)(t)
    return reduce(hcat, [interp(t) for interp in f.interpolators])
end


"""
    times: 1d Array length L
    data: 2d array size (L, ...)
"""
struct LinearInterpolator{T} <: Interpolator
    times::Array{T,1}
    data::Array{T,2}
    t_span::Tuple{T,T}
    function LinearInterpolator(times, data)
        t_span = (times[1], times[end])
        new{typeof(times[1])}(times, data, t_span)
    end
end

function (f::LinearInterpolator)(t)
    idx = findfirst(x -> x >= t, f.times) - 1
    idx == 0 ? idx += 1 : nothing
    t_left = f.times[idx]
    t_delta = f.times[idx+1] - t_left
    θ = (t - t_left) / t_delta
    return (one(t) - θ) * f.data[:, idx] + θ * f.data[:, idx+1]
end




""" Interpolation functions """

function linear_interpolate(
    t::T,
    times::AbstractArray{T,1},
    data::AbstractArray{T,2},
) where {T}
    idx = findfirst(x -> x >= t, times) - 1
    idx == 0 ? idx += 1 : nothing
    t_left = times[idx]
    t_delta = times[idx+1] - t_left
    θ = (t - t_left) / t_delta
    return (one(t) - θ) * data[:, idx] .+ θ * data[:, idx+1]
end

function linear_interpolate(
    t::T,
    times::AbstractArray{T,1},
    data::AbstractArray{T,3},
) where {T}
    idx = findfirst(x -> x >= t, times) - 1
    idx == 0 ? idx += 1 : nothing
    t_left = times[idx]
    t_delta = times[idx+1] - t_left
    θ = (t - t_left) / t_delta
    return (one(t) - θ) * data[:, idx, :] .+ θ * data[:, idx+1, :]
end


""" TODO: add more interpolation types """
# https://github.com/PumasAI/DataInterpolations.jl/blob/master/src/interpolation_methods.jl
# https://github.com/PumasAI/DataInterpolations.jl/blob/master/src/interpolation_caches.jl
