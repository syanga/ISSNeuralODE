using Plots: display
using Serialization
include("../src/StableDynamics.jl")
include("../src/netlist.jl")


demo_config = TrainConfig(
    # metadata
    name = "demo",
    data_path = "data/demo/demo_data.jls",
    # train configuration
    num_epochs = 200,
    p_init_fn = initial_params,
    patience = 10,
    logevery = 3,
    # model configuration
    ode = CTRNN1Canonical(3, 6, 2; σ_u = identity, σ = relu),
    link = AffineCanonical(3, 3),
    mod = ISSCTRNN1(3, 6; enable = true, is_fixed = false, ϵ = 1e-3),
    feed = NoFeedthrough(),
    # DynamicsTrainer arguments
    solver = BS3(),
    adaptive = true,
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = ZygoteVJP()),
    dt = deserialize("data/demo/demo_metadata.jls")["dt"],
    reltol = 1e-4,
    abstol = 1e-6,
    initial_condition = solve_ss,
    loss = Flux.Losses.mse,
    optim = ADAMW(1e-3),
    wdecay = WeightDecay(0.0),
    sched = Exp(λ = 5e0, γ = 1.0),
    validation_split = 0.16,
    test_split = 0.0,
    train_split = 0.84,
    batch_size = 8,
    time_dropout = 0.5,
    time_horizon = 500,
    time_horizon_increment = 50,
    time_horizon_max = 500,
    callback_tasks = [
        generate_netlist_task(
            model_name = "demo",
            num_ports = 2,
            input_config = [("", "V", 1), ("", "V", 2)],
            output_config = [("", "I", 1), ("", "I", 2)],
            time_scale = dataset =
                deserialize("data/demo/demo_metadata.jls")["t_scale_factor"],
            input_scale = deserialize("data/demo/demo_metadata.jls")["u_scale"],
            input_bias = deserialize("data/demo/demo_metadata.jls")["u_bias"],
            output_scale = deserialize("data/demo/demo_metadata.jls")["y_scale"],
            output_bias = deserialize("data/demo/demo_metadata.jls")["y_bias"],
        ),
    ],
)
