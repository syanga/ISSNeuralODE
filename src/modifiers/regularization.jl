using DiffEqFlux, Flux, Zygote

Zygote.@nograd diagm
include("eig.jl")

""" Regularization terms as function of the parameters """

function no_regularization(ode, link, mod, feedthrough, p_ode, p_link, p_mod, p_feed)
    return 0.0
end

""" Regularize magnitude of Lur'e-type Lyapunov function parameters """
function lds_condition_V(ode, link, mod, feedthrough, p_ode, p_link, p_mod, p_feed)
    return mean(mod.get_Ω(p_mod))
end

""" Regularize eigenvalues of stability matrix """
function lds_condition_ctrnn0(ode, link, mod, feedthrough, p_ode, p_link, p_mod, p_feed)
    Ω = mod.get_Ω(p_mod)
    τ = exp.(view(p_ode, ode.index_dict["logτ"]))
    A = reshape(view(p_ode, ode.index_dict["A"]), (ode.state_dim, ode.state_dim))

    Ωmat = Ω.*diagm(ones(length(Ω)))
    mat = 0.5*τ.*(Ω.*A)
    return softplus_eigreg(0.5*τ.*Ωmat*A + 0.5*τ.*A'*Ωmat -  Ω.*diagm(ones(length(Ω))))
end

""" Regularize max eigenvalue of stability matrix """
function lds_condition_ctrnn1(ode, link, mod, feedthrough, p_ode, p_link, p_mod, p_feed)
    Ω = mod.get_Ω(p_mod)
    τ = exp.(view(p_ode, ode.index_dict["logτ"]))
    W = reshape(view(p_ode, ode.index_dict["W"]), ode.state_dim, ode.hidden_dim)
    A = reshape(view(p_ode, ode.index_dict["A"]), ode.hidden_dim, ode.state_dim)

    Ωmat = Ω.*diagm(ones(length(Ω)))
    mat = 0.5*τ.*(Ω.*A)*W
    return softplus_eigreg(mat + mat' - Ωmat)
end
