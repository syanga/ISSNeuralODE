using Serialization
using Parameters
using Setfield
using Dates
using Suppressor
using LinearAlgebra


""" Configuration struct for running experiments """
@with_kw struct TrainConfig
    # metadata
    name::String
    result_path::String = "results"
    data_path::String
    log_path::String = ""
    dtype::Type = Float64
    # assume by default that the system has two threads per core
    blas_threads::Int = Integer(0.5 * length(Sys.cpu_info()))
    # train configuration
    num_epochs::Int
    p_init_fn = initial_params
    patience::Int
    verbose::Bool = true
    logevery::Int = 1
    # model configuration
    ode::O where {O<:Ode}
    link::L where {L<:Link}
    mod::M where {M<:Modifier} = Nominal()
    feed::F where {F<:Feedthrough} = NoFeedthrough()
    # DynamicsTrainer arguments
    solver::Any
    adaptive::Bool = true
    sensealg::Any
    abstol = 1e-6
    reltol = 1e-3
    dt = 1.0
    initial_condition = solve_ss
    loss = Flux.Losses.mse
    regularization = no_regularization
    regularization_coef = 0.0
    optim = ADAMW(dtype(1e-3))
    wdecay = WeightDecay(dtype(0.0))
    sched = Step(λ = dtype(1e-3), γ = dtype(1.0), 1)
    validation_split::Float64 = 0.0
    test_split::Float64 = 0.0
    train_split::Float64 = 1.0
    batch_size::Int
    time_dropout::Float64
    time_horizon::Int
    time_horizon_increment::Int
    time_horizon_max::Int
    callback_tasks = []
end


""" Unpack and run model train configuration """
function run_config(c::TrainConfig; runs = 1)
    @unpack_TrainConfig c

    # set number of threads to be used by BLAS
    LinearAlgebra.BLAS.set_num_threads(blas_threads)

    # display multithreading information
    println(
        "Number of BLAS threads: $(ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
), Number of Julia threads: $(Threads.nthreads())",
    )

    for _ = 1:runs
        # create experiment result path
        savepath = joinpath(result_path, name, Dates.format(now(), "HH_MM_SS_MS"))
        if ~ispath(savepath)
            mkpath(savepath)
        end

        # write config
        serialize(joinpath(savepath, "config.jls"), c)

        # write human readable config as txt file
        config_string = @capture_out show(c)
        open(joinpath(savepath, "config.txt"), "w") do io
            write(io, config_string)
        end

        # read data
        dataset = deserialize(data_path)
        t, u, y = dataset["t"], dataset["u"], dataset["y"]

        # model training
        trainer = DynamicsTrainer(
            ode,
            link,
            mod,
            feed,
            solver,
            sensealg,
            Array{dtype,1}(t),
            Array{dtype,3}(u),
            Array{dtype,3}(y);
            savepath = savepath,
            dtype = dtype,
            validation_split = validation_split,
            test_split = test_split,
            abstol = abstol,
            reltol = reltol,
            dt = dt,
            adaptive = adaptive,
            initial_condition = initial_condition,
            loss = loss,
            regularization = regularization,
            regularization_coef = regularization_coef,
            optim = optim,
            wdecay = wdecay,
            sched = sched,
            batch_size = batch_size,
            callback_tasks = callback_tasks,
        )

        train(
            trainer,
            num_epochs;
            p_init = p_init_fn(trainer; log_path = log_path),
            time_dropout = time_dropout,
            time_horizon = time_horizon,
            time_horizon_increment = time_horizon_increment,
            time_horizon_max = time_horizon_max,
            patience = patience,
            logevery = logevery,
            verbose = verbose,
        )
    end
end
