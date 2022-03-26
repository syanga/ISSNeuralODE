
""" Batch evaluate ODE from initial condition x0 starting from time t0, evaluate at times ty_batch """
function evaluate_ode(
    model,
    link,
    p_ode,
    p_link,
    batch_interpolator,
    x0,
    t0,
    ty_batch,
    solver,
    state_dim,
    output_dim,
    batchsize;
    dynamics_scale = 1.0,
    dt = ty_batch[2] - ty_batch[1],
    adaptive = true,
    abstol = 1e-6,
    reltol = 1e-3,
    sensealg = TrackerAdjoint(),
)
    # construct ODEProblem using the input function
    odeprob = ODEProblem(
        (x, p, t) -> dynamics_scale * model(vcat(x, batch_interpolator(t)), p),
        x0,
        (t0, ty_batch[end]),
        p_ode,
    )

    # trajectory x has shape (state_dim, batchsize, time horizon)
    if adaptive
        x = Array(
            solve(
                odeprob,
                solver;
                saveat = ty_batch,
                abstol = abstol,
                reltol = reltol,
                sensealg = sensealg,
            ),
        )
    else
        x = Array(
            solve(
                odeprob,
                solver;
                dt = dt,
                saveat = ty_batch,
                abstol = abstol,
                reltol = reltol,
                sensealg = sensealg,
            ),
        )
    end

    # prediction ŷ has shape (output_dim, time_horizon, batchsize)
    horizon = length(ty_batch)
    ŷ = permutedims(
        reshape(
            link(reshape(x, state_dim, batchsize * horizon), p_link),
            output_dim,
            batchsize,
            horizon,
        ),
        [1, 3, 2],
    )
end
