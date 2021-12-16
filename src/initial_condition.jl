using NonlinearSolve

""" Set the initial condition: rootfinding for finding the steady state solution """
function solve_ss(model, u0, p_ode, state_dim, batchsize; u0_guess=zeros(state_dim, batchsize), abstol=1e-10)
    # backward pass with matrix valued steady state does not seem to work
    # ssprob = SteadyStateProblem((x,p,t) -> model(vcat(reshape(x, state_dim, batchsize), u0), p), u0_guess, p_ode)
    # x0 = reshape(Array(solve(ssprob, SSRootfind(nlsolve=(f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,ftol=abstol, method=method);res.zero)); abstol=abstol)), state_dim, batchsize)
    # return x0
    ssprob = NonlinearProblem{false}((x,p) -> vec(model(vcat(reshape(x, state_dim, batchsize), u0), p)), vec(u0_guess), p_ode)
    return reshape(Array(solve(ssprob, NewtonRaphson(), tol=abstol)), state_dim, batchsize)
end

""" Set the initial condition: rootfinding for finding the steady state solution, but do not backprop through this """
function solve_ss_dropgrad(model, u0, p_ode, state_dim, batchsize; u0_guess=zeros(state_dim, batchsize), method=:newton, abstol=1e-10)
    x0 = u0_guess
    Zygote.ignore() do
        # ssprob = SteadyStateProblem((x,p,t) -> model(vcat(reshape(x, state_dim, batchsize), u0), p), u0_guess, p_ode)
        # x0 = reshape(Array(solve(ssprob, SSRootfind(nlsolve=(f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,ftol=abstol, method=method);res.zero)); abstol=abstol)), state_dim, batchsize)
        ssprob = NonlinearProblem{false}((x,p) -> vec(model(vcat(reshape(x, state_dim, batchsize), u0), p)), vec(u0_guess), p_ode)
        x0 = reshape(Array(solve(ssprob, NewtonRaphson(), tol=abstol)), state_dim, batchsize)
    end
    return x0
end

""" Set the initial condition: fixed value, default Float64 zeros """
function fixed_ic(model, u0, p_ode, state_dim, batchsize; u0_guess=zeros(state_dim, batchsize), method=:newton, abstol=1e-8)
    return u0_guess
end
