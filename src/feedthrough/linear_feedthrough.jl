using LinearAlgebra

""" y = Du
LinearFeedthrough """
struct LinearFeedthrough{F} <: Feedthrough
    input_dim::Int
    output_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    initial_params::F
    function LinearFeedthrough(input_dim::Int, output_dim::Int; initW = Flux.glorot_uniform)
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["D"] = i:i+output_dim*input_dim-1
        i += output_dim * input_dim

        # initialize initial parameters
        initial_params() = vec(initW(output_dim, input_dim))

        new{typeof(initial_params)}(
            input_dim,
            output_dim,
            i - 1,
            index_dict,
            initial_params,
        )
    end
end

function unpack_params(f::LinearFeedthrough, p)
    D = reshape(p[f.index_dict["D"]], f.output_dim, f.input_dim)
    return D
end

function (f::LinearFeedthrough)(ȳ, u, p)
    D = reshape(p[f.index_dict["D"]], f.output_dim, f.input_dim)
    Du = reshape(D * reshape(u, size(u, 1), :), :, size(u)[2:end]...)
    return ȳ .+ Du
end


""" y = [D 0]u where D is square diagonal
ResistiveFeedthrough """
struct ResistiveFeedthrough{F} <: Feedthrough
    input_dim::Int
    output_dim::Int
    paramsum::Int
    index_dict::Dict{String,UnitRange{Int}}
    initial_params::F
    function ResistiveFeedthrough(
        input_dim::Int,
        output_dim::Int;
        initW = Flux.glorot_uniform,
    )
        # compute indices
        i = 1
        index_dict = Dict{String,UnitRange{Int}}()
        index_dict["D"] = i:i+output_dim-1
        i += output_dim

        # initialize initial parameters
        initial_params() = vec(initW(output_dim))

        new{typeof(initial_params)}(
            input_dim,
            output_dim,
            i - 1,
            index_dict,
            initial_params,
        )
    end
end

function unpack_params(f::ResistiveFeedthrough, p)
    D = hcat(
        diagm(p[f.index_dict["D"]] .^ 2),
        zeros(f.output_dim, max(0, f.input_dim - f.output_dim)),
    )
    return D
end

function (f::ResistiveFeedthrough)(ȳ, u, p)
    D = p[f.index_dict["D"]] .^ 2
    ul = @view u[1:f.output_dim, :, :]
    Du = reshape(D .* reshape(ul, size(ul, 1), :), :, size(ul)[2:end]...)
    return ȳ .+ Du
end
