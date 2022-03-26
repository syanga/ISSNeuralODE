using DiffEqFlux, Flux

Zygote.@nograd diagm
include("eig.jl")

""" LDS condition for CTRNN1
        Enforce Ω(I - τAW) + (I - τW'A')Ω ≻ 0,
        parametrize sqrt(Ω) """
struct ISSCTRNN1{P,Q1,Q2,Q3,F} <: Modifier
    name::String
    param_size::Int
    state_dim::Int
    hidden_dim::Int
    initial_params::P
    get_Ω::Q1
    get_sqrtΩ::Q2
    get_sqrtΩinv::Q3
    ϵ::F
    lb::F
    logτ_idx::UnitRange{Int}
    W_idx::UnitRange{Int}
    A_idx::UnitRange{Int}
    enable::Bool
    scale_W::Bool
    function ISSCTRNN1(
        state_dim,
        hidden_dim;
        enable = true,
        is_fixed = false,
        scale_W = false,
        initb = Flux.zeros,
        ϵ = 1e-3,
        shift = -4.0,
    )
        # initial parameters
        initial_params() = vec(initb(hidden_dim))

        # function to fetch parameters: if fixed, ignore parameters
        if is_fixed
            Ω_init = ones(hidden_dim)
            get_Ω = (p) -> Ω_init
            get_sqrtΩ = (p) -> Ω_init
            get_sqrtΩinv = (p) -> Ω_init
        else
            # get_sqrtΩ = (p) -> 0.1 .+ log.(1.0 .+ p .^ 2) #1.0.+exp.(-p.+shift)
            get_sqrtΩ = (p) -> softmax(p)
            get_sqrtΩinv = (p) -> 1.0 ./ get_sqrtΩ(p)  #sigmoid.(p.-shift)

            get_Ω = (p) -> get_sqrtΩ(p) .^ 2
            # get_sqrtΩ = (p) -> 1.0.+exp.(-p.+shift)
            # get_sqrtΩinv = (p) -> sigmoid.(p.-shift)
        end

        # parameter indices: assume computed same way as by CTRNN1
        i = 1
        logτ_idx = i:i
        i += 1
        W_idx = i:i+state_dim*hidden_dim-1
        i += state_dim * hidden_dim
        A_idx = i:i+hidden_dim*state_dim-1

        new{
            typeof(initial_params),
            typeof(get_Ω),
            typeof(get_sqrtΩ),
            typeof(get_sqrtΩinv),
            typeof(ϵ),
        }(
            "ISSCTRNN1",
            hidden_dim,
            state_dim,
            hidden_dim,
            initial_params,
            get_Ω,
            get_sqrtΩ,
            get_sqrtΩinv,
            ϵ,
            shift,
            logτ_idx,
            W_idx,
            A_idx,
            enable,
            scale_W,
        )
    end
end


""" Modify p_sub to guarantee stability """
function (f::ISSCTRNN1)(p_sub, p_link, p; T = Float64, rhs = (one(T) - T(f.ϵ)))
    if f.enable
        sqrtΩ = f.get_sqrtΩ(p)
        invsqrtΩ = f.get_sqrtΩinv(p)

        τ = exp.(@view p_sub[f.logτ_idx])
        W = reshape(view(p_sub, f.W_idx), f.state_dim, f.hidden_dim)
        A = reshape(view(p_sub, f.A_idx), f.hidden_dim, f.state_dim)

        mat = T(0.5) * τ .* (sqrtΩ .* A) * (W' .* invsqrtΩ)'
        factor = one(T) / (one(T) + relu(maxeig(mat + mat') - one(T) + T(f.ϵ)))

        if f.scale_W
            _p_sub = vcat(
                p_sub[1],
                factor * view(p_sub, f.W_idx),
                view(p_sub, f.A_idx),
                p_sub[f.A_idx[end]+1:end],
            )
        else
            _p_sub = vcat(
                p_sub[1],
                view(p_sub, f.W_idx),
                factor * view(p_sub, f.A_idx),
                p_sub[f.A_idx[end]+1:end],
            )
        end

        return _p_sub, p_link
    else
        return p_sub, p_link
    end
end


"""  Debug: Computes the largest eigenvalue to ensure Ω(τAW - I) + (τW'A' - I)Ω ≺ 0 """
function debug(f::ISSCTRNN1, p_sub, p_link, p)
    Ω = f.get_Ω(p)
    τ = exp.(view(p_sub, f.logτ_idx))
    W = reshape(view(p_sub, f.W_idx), f.state_dim, f.hidden_dim)
    A = reshape(view(p_sub, f.A_idx), f.hidden_dim, f.state_dim)

    mat = Ω .* τ .* A * W - diagm(Ω)
    return maxeig(mat + mat')
end


##############################################################################
##############################################################################
##############################################################################


""" LDS condition for CTRNN1
        Enforce Ω(I - τAW) + (I - τW'A')Ω ≻ 0,
        parametrize sqrt(Ω) """
struct ISSCTRNN1Alt{P,Q1,Q2,Q3,F} <: Modifier
    name::String
    param_size::Int
    state_dim::Int
    hidden_dim::Int
    initial_params::P
    get_Ω::Q1
    get_sqrtΩ::Q2
    get_sqrtΩinv::Q3
    ϵ::F
    lb::F
    logτ_idx::UnitRange{Int}
    W_idx::UnitRange{Int}
    A_idx::UnitRange{Int}
    enable::Bool
    scale_W::Bool
    function ISSCTRNN1Alt(
        state_dim,
        hidden_dim;
        enable = true,
        is_fixed = false,
        scale_W = false,
        initb = Flux.zeros,
        ϵ = 1e-3,
        lb = -3.0,
    )
        # initial parameters
        initial_params() = vec(initb(hidden_dim))

        # function to fetch parameters: if fixed, ignore parameters
        if is_fixed
            Ω_init = ones(hidden_dim)
            get_Ω = (p) -> Ω_init
            get_sqrtΩ = (p) -> Ω_init
            get_sqrtΩinv = (p) -> Ω_init
        else
            get_sqrtΩ = (p) -> 0.5 .+ exp.(p) #log.(1.0 .+ p.^2)
            get_Ω = (p) -> get_sqrtΩ(p) .^ 2
            get_sqrtΩinv = (p) -> 1.0 ./ get_sqrtΩ(p)
        end

        # parameter indices: assume computed same way as by CTRNN1
        i = 1
        logτ_idx = i:i
        i += 1
        W_idx = i:i+state_dim*hidden_dim-1
        i += state_dim * hidden_dim
        A_idx = i:i+hidden_dim*state_dim-1

        new{
            typeof(initial_params),
            typeof(get_Ω),
            typeof(get_sqrtΩ),
            typeof(get_sqrtΩinv),
            typeof(ϵ),
        }(
            "ISSCTRNN1Alt",
            hidden_dim,
            state_dim,
            hidden_dim,
            initial_params,
            get_Ω,
            get_sqrtΩ,
            get_sqrtΩinv,
            ϵ,
            lb,
            logτ_idx,
            W_idx,
            A_idx,
            enable,
            scale_W,
        )
    end
end


""" Modify p_sub to guarantee stability """
function (f::ISSCTRNN1Alt)(p_sub, p_link, p; T = Float64, rhs = (one(T) - T(f.ϵ)))
    if f.enable
        sqrtΩ = f.get_sqrtΩ(p)
        # Ω = get_Ω(p)

        τ = exp.(@view p_sub[f.logτ_idx])
        W = reshape(view(p_sub, f.W_idx), f.state_dim, f.hidden_dim)
        A = reshape(view(p_sub, f.A_idx), f.hidden_dim, f.state_dim)

        # mat = T(0.5)*τ.*(sqrtΩ.*A)*(W'.*sqrtΩ)'
        mat = T(0.5) * τ .* (sqrtΩ .* A) * W
        factor = one(T) / (one(T) + relu(maxeig(mat + mat') - one(T) + T(f.ϵ)))

        if f.scale_W
            _p_sub = vcat(
                p_sub[1],
                factor * vec((W' .* sqrtΩ)'),
                view(p_sub, f.A_idx),
                p_sub[f.A_idx[end]+1:end],
            )
        else
            _p_sub = vcat(
                p_sub[1],
                vec((W' .* sqrtΩ)'),
                factor * view(p_sub, f.A_idx),
                p_sub[f.A_idx[end]+1:end],
            )
        end

        return _p_sub, p_link
    else
        return p_sub, p_link
    end
end


"""  Debug: Computes the largest eigenvalue to ensure Ω(τAW - I) + (τW'A' - I)Ω ≺ 0 """
function debug(f::ISSCTRNN1Alt, p_sub, p_link, p)
    Ω = f.get_Ω(p)
    τ = exp.(view(p_sub, f.logτ_idx))
    W = reshape(view(p_sub, f.W_idx), f.state_dim, f.hidden_dim)
    A = reshape(view(p_sub, f.A_idx), f.hidden_dim, f.state_dim)

    mat = Ω .* τ .* A * W - diagm(Ω)
    return maxeig(mat + mat')
end
