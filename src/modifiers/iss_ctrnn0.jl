using DiffEqFlux, Flux, Zygote

Zygote.@nograd diagm
include("eig.jl")

""" LDS condition for CTRNN0
        Enforce Ω(I - τA) + (I - τA')Ω ≻ 0,
        parametrize sqrt(Ω) """
struct ISSCTRNN0{P,Q1,Q2,Q3,F} <: Modifier
    name::String
    param_size::Int
    state_dim::Int
    initial_params::P
    get_Ω::Q1
    get_sqrtΩ::Q2
    get_sqrtΩinv::Q3
    ϵ::F
    lb::F
    logτ_idx::UnitRange{Int}
    A_idx::UnitRange{Int}
    enable::Bool
    function ISSCTRNN0(
        state_dim;
        enable = true,
        is_fixed = false,
        initb = Flux.zeros,
        ϵ = 1e-3,
        scale = 0.5,
        shift = -3.0,
    )
        # initial parameters
        initial_params() = vec(initb(state_dim))

        # function to fetch parameters: if fixed, ignore parameters
        if is_fixed
            Ω_init = ones(state_dim)
            get_Ω = (p) -> Ω_init
            get_sqrtΩ = (p) -> Ω_init
            get_sqrtΩinv = (p) -> Ω_init
        else
            # get_sqrtΩ = (p) -> 0.1 .+ log.(1.0 .+ p .^ 2) #Flux.huber_loss(zeros(state_dim), p, agg=identity)
            get_sqrtΩ = (p) -> softmax(p)
            get_sqrtΩinv = (p) -> 1.0 ./ get_sqrtΩ(p)

            # get_sqrtΩ = (p) -> scale*(1.0.+exp.(-p.+shift))
            # get_sqrtΩinv = (p) -> sigmoid.(p.-shift)/scale
            get_Ω = (p) -> get_sqrtΩ(p) .^ 2
        end

        # parameter indices: assume computed same way as by CTRNN0
        i = 1
        logτ_idx = i:i
        i += 1
        A_idx = i:i+state_dim*state_dim-1

        # logτ_idx = subsystem.index_dict["logτ"]
        # A_idx = subsystem.index_dict["A"]

        new{
            typeof(initial_params),
            typeof(get_Ω),
            typeof(get_sqrtΩ),
            typeof(get_sqrtΩinv),
            typeof(ϵ),
        }(
            "ISSCTRNN0",
            state_dim,
            state_dim,
            initial_params,
            get_Ω,
            get_sqrtΩ,
            get_sqrtΩinv,
            ϵ,
            shift,
            logτ_idx,
            A_idx,
            enable,
        )
    end
end

""" Modify p_sub to guarantee stability """
function (f::ISSCTRNN0)(p_sub, p_link, p; T = Float64, rhs = T(f.ϵ))
    if f.enable
        # Ω = f.get_Ω(p)
        sqrtΩ = f.get_sqrtΩ(p)
        invsqrtΩ = f.get_sqrtΩinv(p)

        τ = exp.(@view p_sub[f.logτ_idx])
        A = reshape(view(p_sub, f.A_idx), f.state_dim, f.state_dim)

        mat = T(0.5) * τ .* ((sqrtΩ .* A)' .* invsqrtΩ)'
        factor = one(T) / (one(T) + relu(maxeig(mat + mat') - one(T) + T(f.ϵ)))

        _p_sub = vcat(p_sub[1], factor * vec(A), p_sub[f.A_idx[end]+1:end])

        return _p_sub, p_link
    else
        return p_sub, p_link
    end
end

"""  Debug: Computes the largest eigenvalue to ensure Ω(τA - I) + (τA' - I)Ω ≺ 0 """
function debug(f::ISSCTRNN0, p_sub, p_link, p)
    Ω = f.get_Ω(p)

    τ = exp.(@view p_sub[f.logτ_idx])
    A = reshape(view(p_sub, f.A_idx), f.state_dim, f.state_dim)

    mat = τ .* Ω .* A - diagm(Ω)
    return maxeig(mat + mat')
end



##############################################################################
##############################################################################
##############################################################################



""" LDS condition for CTRNN0
        Enforce Ω(I - τA) + (I - τA')Ω ≻ 0,
        parametrize sqrt(Ω) """
struct ISSCTRNN0Alt{P,Q1,Q2,Q3,F} <: Modifier
    name::String
    param_size::Int
    state_dim::Int
    initial_params::P
    get_Ω::Q1
    get_sqrtΩ::Q2
    get_sqrtΩinv::Q3
    ϵ::F
    lb::F
    logτ_idx::UnitRange{Int}
    A_idx::UnitRange{Int}
    enable::Bool
    function ISSCTRNN0Alt(
        state_dim;
        enable = true,
        is_fixed = false,
        initb = Flux.zeros,
        ϵ = 1e-3,
        lb = -3.0,
    )
        # initial parameters
        initial_params() = vec(initb(state_dim))

        # function to fetch parameters: if fixed, ignore parameters
        if is_fixed
            Ω_init = ones(state_dim)
            get_Ω = (p) -> Ω_init
            get_sqrtΩ = (p) -> Ω_init
            get_sqrtΩinv = (p) -> Ω_init
        else
            get_sqrtΩ = (p) -> 0.5 .+ exp.(p) #log.(1.0.+p.^2)
            get_sqrtΩinv = (p) -> 1.0 ./ get_sqrtΩ(p)
            get_Ω = (p) -> get_sqrtΩ(p) .^ 2
        end

        # parameter indices: assume computed same way as by CTRNN0
        i = 1
        logτ_idx = i:i
        i += 1
        A_idx = i:i+state_dim*state_dim-1

        # logτ_idx = subsystem.index_dict["logτ"]
        # A_idx = subsystem.index_dict["A"]

        new{
            typeof(initial_params),
            typeof(get_Ω),
            typeof(get_sqrtΩ),
            typeof(get_sqrtΩinv),
            typeof(ϵ),
        }(
            "ISSCTRNN0Alt",
            state_dim,
            state_dim,
            initial_params,
            get_Ω,
            get_sqrtΩ,
            get_sqrtΩinv,
            ϵ,
            lb,
            logτ_idx,
            A_idx,
            enable,
        )
    end
end

""" Modify p_sub to guarantee stability """
function (f::ISSCTRNN0Alt)(p_sub, p_link, p; T = Float64, rhs = T(f.ϵ))
    if f.enable
        sqrtΩ = f.get_sqrtΩ(p)
        Ω = f.get_Ω(p)

        τ = exp.(@view p_sub[f.logτ_idx])
        A = reshape(view(p_sub, f.A_idx), f.state_dim, f.state_dim)

        # mat = T(0.5)*τ.*((sqrtΩ.*A)'.*sqrtΩ)'
        mat = T(0.5) * τ .* sqrtΩ .* A
        factor = one(T) / (one(T) + relu(maxeig(mat + mat') - one(T) + T(f.ϵ)))

        _p_sub = vcat(p_sub[1], factor * vec((A' .* sqrtΩ)'), p_sub[f.A_idx[end]+1:end])

        return _p_sub, p_link
    else
        return p_sub, p_link
    end
end

"""  Debug: Computes the largest eigenvalue to ensure Ω(τA - I) + (τA' - I)Ω ≺ 0 """
function debug(f::ISSCTRNN0Alt, p_sub, p_link, p)
    Ω = f.get_Ω(p)

    τ = exp.(@view p_sub[f.logτ_idx])
    A = reshape(view(p_sub, f.A_idx), f.state_dim, f.state_dim)

    mat = τ .* Ω .* A - diagm(Ω)
    return maxeig(mat + mat')
end
