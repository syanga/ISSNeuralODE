using Serialization
using Plots
using Statistics
using Random

# # make plots with no gui
# ENV["GKSwstype"]="nul"

""" Manage model training 
    t: time steps, size(time_horizon,)
    u: inputs, size (input_dim, time_horizon, data_size) 
    y: outputs, size (output_dim, time_horizon, data_size) """
mutable struct DynamicsTrainer{T,O,L,M,F,S,SA,OPT,SC,I,J,R,C,D}
    # models, paramlengths, and other model parameters
    ode::O
    link::L
    mod::M
    feedthrough::F
    pl_ode::Int
    pl_link::Int
    pl_mod::Int
    pl_feed::Int
    state_dim::Int
    input_dim::Int
    output_dim::Int

    # solver and sensealg setup
    solver::S
    sensealg::SA
    abstol::T
    reltol::T
    dt::T
    adaptive::Bool

    # set initial condition
    initial_condition::I

    # optimizer and learning rate schedule
    opt::OPT
    schedule::SC

    # time points
    t0::T
    t::Array{T,1}
    time_batch::Int
    time_idx::Array{Int,1}

    # time horizon control
    time_horizon::Int
    time_horizon_increment::Int
    time_horizon_max::Int
    time_horizons::Array{Int,1}

    # parameters and best parameters
    p::Array{T,1}
    p_opt::Array{T,1}
    loss_opt::T
    not_improved::Int
    patience::Int

    # variable values
    iteration::Int
    epoch::Int
    iteration_time::T
    epoch_time::T

    # keep track of and print training progress
    callback_tasks::C
    train_losses::Array{T,1}
    epoch_train_losses::Array{T,1}
    valid_losses::Array{T,1}
    train_regularizations::Array{T,1}
    epoch_train_regularizations::Array{T,1}
    valid_regularizations::Array{T,1}
    learn_rates::Array{T,1}
    epoch_times::Array{T,1}
    mod_stats_pre::Array{T,1}
    mod_stats_post::Array{T,1}
    verbose::Bool
    logevery::Int
    savepath::String

    # loss function and dataloaders
    loss::J
    regularization::R
    regularization_coef::T
    train_data_loader::D
    valid_data_loader::D
    test_data_loader::D
    epoch_size::Int

    function DynamicsTrainer(
        ode,
        link,
        mod,
        feedthrough,
        solver,
        sensealg,
        t::Array{T,1},
        u::Array{T,3},
        y::Array{T,3};
        savepath = "",
        dtype = Float64,
        validation_split = 0.0,
        test_split = 0.0,
        train_split = 1.0 - validation_split - test_split,
        abstol = dtype(1e-6),
        reltol = dtype(1e-3),
        dt = dtype(1.0),
        adaptive = false,
        initial_condition = solve_ss,
        loss = Flux.Losses.mse,
        regularization = no_regularization,
        regularization_coef = dtype(0.0),
        optim = ADAMW(dtype(1e-3)),
        wdecay = WeightDecay(dtype(0.0)),
        sched = Step(λ = dtype(1e-3), γ = dtype(1.0), 1),
        batch_size = 1,
        time_dropout = 0.0,
        time_horizon = size(u)[2],
        time_horizon_increment = 0,
        time_horizon_max = time_horizon,
        callback_tasks = [],
    ) where {T}
        # metadata
        pl_ode = DiffEqFlux.paramlength(ode)
        pl_link = DiffEqFlux.paramlength(link)
        pl_mod = DiffEqFlux.paramlength(mod)
        pl_feed = DiffEqFlux.paramlength(feedthrough)
        state_dim = ode.state_dim
        input_dim = ode.input_dim
        output_dim = link.output_dim

        # optimization
        opt = Flux.Optimiser(optim, wdecay)
        schedule = Stateful(sched)

        # data time step and time minibatching
        t0 = t[1]
        time_batch = Integer(floor(time_horizon * (1.0 - time_dropout)))
        time_idx = collect([1:1:time_batch;])

        # model parameters
        # p = vcat(Array{dtype,1}(initial_params(ode)), 
        #         Array{dtype,1}(initial_params(link)), 
        #         Array{dtype,1}(initial_params(mod)))
        p = zeros(paramlength(ode) + paramlength(link) + paramlength(mod))
        p_opt = copy(p)
        loss_opt = dtype(Inf)
        not_improved = 0
        patience = 1
        logevery = 1

        # counters and timers
        iteration = 0
        epoch = 1
        iteration_time = time()
        epoch_time = time()

        # keep track of and print training progress
        train_losses = Array{dtype,1}()
        epoch_train_losses = Array{dtype,1}()
        valid_losses = Array{dtype,1}()
        train_regularizations = Array{dtype,1}()
        epoch_train_regularizations = Array{dtype,1}()
        valid_regularizations = Array{dtype,1}()
        learn_rates = Array{dtype,1}()
        epoch_times = Array{dtype,1}()
        mod_stats_pre = Array{dtype,1}()
        mod_stats_post = Array{dtype,1}()
        time_horizons = Array{Int,1}()
        verbose = true
        savepath = savepath

        # loss function and dataloaders
        train_dl, valid_dl, test_dl = create_dataloaders(
            u,
            y,
            batch_size;
            validation_split = validation_split,
            test_split = test_split,
            train_split = train_split,
        )
        epoch_size = length(train_dl)

        new{
            dtype,
            typeof(ode),
            typeof(link),
            typeof(mod),
            typeof(feedthrough),
            typeof(solver),
            typeof(sensealg),
            typeof(opt),
            typeof(schedule),
            typeof(initial_condition),
            typeof(loss),
            typeof(regularization),
            typeof(callback_tasks),
            typeof(train_dl),
        }(
            ode,
            link,
            mod,
            feedthrough,
            pl_ode,
            pl_link,
            pl_mod,
            pl_feed,
            state_dim,
            input_dim,
            output_dim,
            solver,
            sensealg,
            abstol,
            reltol,
            dt,
            adaptive,
            initial_condition,
            opt,
            schedule,
            t0,
            t,
            time_batch,
            time_idx,
            time_horizon,
            time_horizon_increment,
            time_horizon_max,
            time_horizons,
            p,
            p_opt,
            loss_opt,
            not_improved,
            patience,
            iteration,
            epoch,
            iteration_time,
            epoch_time,
            callback_tasks,
            train_losses,
            epoch_train_losses,
            valid_losses,
            train_regularizations,
            epoch_train_regularizations,
            valid_regularizations,
            learn_rates,
            epoch_times,
            mod_stats_pre,
            mod_stats_post,
            verbose,
            logevery,
            savepath,
            loss,
            regularization,
            regularization_coef,
            train_dl,
            valid_dl,
            test_dl,
            epoch_size,
        )
    end
end

""" Run training loop 
    t: Array of time values corresponding to train and validation dataloaders 
    train_data_loader: dataloader given data with shape (input_dim, time_dim, data_size)
    valid_data_loader: dataloader given data with shape (input_dim, time_dim, data_size) """
function train(
    f::DynamicsTrainer,
    num_epochs;
    p_init = initial_params(f),
    time_dropout = 0.0,
    time_horizon = f.time_horizon,
    time_horizon_increment = 0,
    time_horizon_max = f.time_horizon_max,
    patience = 1,
    logevery = 1,
    verbose = true,
)
    # time dropout
    f.time_batch = Integer(floor(time_horizon * (1.0 - time_dropout)))
    f.time_idx = collect([1:1:f.time_batch;])

    # time horizons
    f.time_horizon = time_horizon
    f.time_horizon_increment = time_horizon_increment
    f.time_horizon_max = time_horizon_max

    # set initial parameters
    f.p = copy(p_init)
    f.p_opt = copy(p_init)

    f.loss_opt = typeof(f.loss_opt)(Inf)
    f.not_improved = 0
    f.patience = patience
    f.logevery = logevery

    f.iteration = 0
    f.epoch = 1

    empty!(f.train_losses)
    empty!(f.epoch_train_losses)
    empty!(f.valid_losses)
    empty!(f.train_regularizations)
    empty!(f.epoch_train_regularizations)
    empty!(f.valid_regularizations)
    empty!(f.learn_rates)
    empty!(f.epoch_times)
    empty!(f.time_horizons)
    empty!(f.mod_stats_pre)
    empty!(f.mod_stats_post)
    f.verbose = verbose

    # initialize timers
    f.iteration_time = time()
    f.epoch_time = time()

    # run training loop
    Flux.train!(
        (u, y) -> forward(f, u, y),
        Flux.params(f.p),
        ncycle(f.train_data_loader, num_epochs),
        f.opt,
        cb = () -> callback(f),
    )
    return f.p
end

""" Generate default initial model parameters """
function initial_params(f::DynamicsTrainer; log_path = "")
    if log_path == ""
        p_ode = initial_params(f.ode)
        p_link = initial_params(f.link)
        p_mod = initial_params(f.mod)
        p_feed = initial_params(f.feedthrough)
    else
        logdict = deserialize(log_path)
        p_ode = logdict["p_ode"]
        p_link = logdict["p_link"]
        p_mod = logdict["p_mod"]
        p_feed = logdict["p_feed"]
    end
    return typeof(f.p)(vcat(p_ode, p_link, p_mod, p_feed))
end

""" Extract parameters """
function extract_params(f::DynamicsTrainer)
    _p_ode = @views f.p[1:f.pl_ode]
    _p_link = @views f.p[f.pl_ode+1:f.pl_ode+f.pl_link]
    p_mod = @views f.p[f.pl_ode+f.pl_link+1:f.pl_ode+f.pl_link+f.pl_mod]
    p_feed = @views f.p[f.pl_ode+f.pl_link+f.pl_mod+1:end]
    p_ode, p_link = f.mod(_p_ode, _p_link, p_mod)
    return _p_ode, _p_link, p_mod, p_feed, p_ode, p_link
end

""" Create dataloader using uniformly sampled input data """
function create_dataloaders(
    u,
    y,
    batch_size;
    validation_split = 0.0,
    test_split = 0.0,
    train_split = 1.0 - validation_split - test_split,
    dataset_size = size(y)[3],
)
    # set up data loaders: use first validation_split fraction of samples as the validation set
    n_total = size(u, 3)
    n_train_samples = Integer(floor(dataset_size * train_split))
    n_valid_samples = Integer(floor(dataset_size * validation_split))
    n_test_samples = Integer(floor(dataset_size * test_split))

    # randomly subsample
    shuffled_idx = randperm(n_total)
    valid_idx = shuffled_idx[1:n_valid_samples]
    test_idx = shuffled_idx[n_valid_samples+1:n_valid_samples+n_test_samples]
    train_idx =
        shuffled_idx[n_valid_samples+n_test_samples+1:n_valid_samples+n_test_samples+n_train_samples]

    # create validation data loader using first n_valid_samples samples
    if n_valid_samples > 0
        validation_data_loader = Flux.Data.DataLoader(
            (u[:, :, valid_idx], y[:, :, valid_idx]);
            batchsize = batch_size,
            shuffle = true,
        )
    else
        validation_data_loader =
            Flux.Data.DataLoader((zeros(0, 0, 0), zeros(0, 0, 0)), batchsize = batch_size)
    end

    # create test data loader using next n_test_samples
    if n_test_samples > 0
        test_data_loader = Flux.Data.DataLoader(
            (u[:, :, test_idx], y[:, :, test_idx]);
            batchsize = batch_size,
            shuffle = true,
        )
    else
        test_data_loader =
            Flux.Data.DataLoader((zeros(0, 0, 0), zeros(0, 0, 0)), batchsize = batch_size)
    end

    # create training data loader using rest of data
    if n_train_samples > 0
        train_data_loader = Flux.Data.DataLoader(
            (u[:, :, train_idx], y[:, :, train_idx]);
            batchsize = batch_size,
            shuffle = true,
        )
    else
        test_data_loader =
            Flux.Data.DataLoader((zeros(0, 0, 0), zeros(0, 0, 0)), batchsize = batch_size)
    end

    return train_data_loader, validation_data_loader, test_data_loader
end

""" Takes dataloader output and compute model output """
function predict(f::DynamicsTrainer, u; full = false)
    # extract parameters & apply modifier
    _, _, _, p_feed, p_ode, p_link = extract_params(f)

    # function to evaluate convert batch of interpolants
    batch_interpolator(s) =
        @views linear_interpolate(s, f.t, full ? u : u[:, 1:f.time_horizon, :])

    # sample random times on which to compute loss
    Zygote.ignore() do
        f.time_idx = vcat(1, sort((randperm(f.time_horizon - 1).+1)[1:f.time_batch-1]))
    end
    t_batch = full ? f.t : @view f.t[f.time_idx]
    curr_batchsize = size(u)[3]

    # steady state solve with initial input for initial condition
    u0 = u[:, 1, :]
    x0 = (f.initial_condition)(
        f.ode,
        u0,
        p_ode,
        f.state_dim,
        curr_batchsize;
        u0_guess = typeof(u0)(zeros(f.state_dim, curr_batchsize)),
    )

    # compute predicted trajectory
    ȳ = evaluate_ode(
        f.ode,
        f.link,
        p_ode,
        p_link,
        batch_interpolator,
        x0,
        f.t0,
        t_batch,
        f.solver,
        f.state_dim,
        f.output_dim,
        curr_batchsize;
        dynamics_scale = typeof(f.t0)(1.0),
        dt = f.dt,
        adaptive = f.adaptive,
        abstol = f.abstol,
        reltol = f.reltol,
        sensealg = f.sensealg,
    )

    # add feedthrough
    t_feed = @views full ? collect(1:size(u, 2)) : f.time_idx
    ŷ = f.feedthrough(ȳ, u[:, t_feed, :], p_feed)
    return ŷ
end

""" Takes dataloader output and compute scalar loss """
function forward(f::DynamicsTrainer, u, y)
    # compute model output
    ŷ = predict(f, u)

    # compute loss
    if f.regularization_coef > 0.0
        _p_ode, _p_link, p_mod, p_feed, _, _ = extract_params(f)

        reg =
            f.regularization_coef * (f.regularization)(
                f.ode,
                f.link,
                f.mod,
                f.feedthrough,
                _p_ode,
                _p_link,
                p_mod,
                p_feed,
            )
        batch_loss = @views (f.loss)(ŷ, y[:, f.time_idx, :]) + reg

    else
        batch_loss = @views (f.loss)(ŷ, y[:, f.time_idx, :])
        reg = 0.0
    end

    # save train loss and count iterations
    Zygote.ignore() do
        push!(f.train_losses, batch_loss - reg)
        push!(f.train_regularizations, reg)
    end

    return batch_loss
end

""" Takes dataloader output and compute scalar loss and regularization term in validation mode """
function forward_validation(f::DynamicsTrainer, u, y)
    # compute model output
    ŷ = predict(f, u)

    # compute loss
    _p_ode, _p_link, p_mod, p_feed, _, _ = extract_params(f)
    reg =
        f.regularization_coef * (f.regularization)(
            f.ode,
            f.link,
            f.mod,
            f.feedthrough,
            _p_ode,
            _p_link,
            p_mod,
            p_feed,
        )
    batch_loss = @views (f.loss)(ŷ, y[:, f.time_idx, :]) + reg
    return batch_loss, reg
end

""" Run once every training iteration """
function callback(f::DynamicsTrainer)::Bool
    Zygote.ignore() do
        iteration_time = time() - f.iteration_time
        f.iteration += 1

        # print progress
        if f.verbose
            print(
                "Epoch progress: $(f.iteration%f.epoch_size)/$(f.epoch_size), time: $(round(iteration_time, digits=2)) s., loss: " *
                string(f.train_losses[end]) *
                "\r",
            )
        end

        # do every epoch
        if f.iteration % f.epoch_size == 0 && f.iteration > 0
            # compute epoch time
            epoch_time = time() - f.epoch_time
            push!(f.epoch_times, epoch_time)

            # since first epoch time can be exceptionally long due to compilation time, replace first time with second 
            if length(f.epoch_times) == 2
                f.epoch_times[1] = f.epoch_times[2]
            end

            # update learning rate based on scheduler - opt has type Flux.Optimiser
            f.opt[1][1].eta = next!(f.schedule)
            push!(f.learn_rates, f.opt[1][1].eta)

            # compute epoch train loss
            epoch_train_loss = mean(f.train_losses[end-f.epoch_size+1:end])
            empty!(f.train_losses)
            push!(f.epoch_train_losses, epoch_train_loss)

            # compute epoch train regularizations
            epoch_train_regularization =
                mean(f.train_regularizations[end-f.epoch_size+1:end])
            empty!(f.train_regularizations)
            push!(f.epoch_train_regularizations, epoch_train_regularization)

            # compute epoch validation loss
            epoch_valid_loss = 0.0
            epoch_reg_valid = 0.0
            if f.valid_data_loader.batchsize > 0
                for (u_val, y_val) in f.valid_data_loader
                    batch_loss_vaid, reg_valid = forward_validation(f, u_val, y_val)
                    epoch_valid_loss += batch_loss_vaid / length(f.valid_data_loader)
                    epoch_reg_valid += reg_valid / length(f.valid_data_loader)
                end
                push!(f.valid_losses, epoch_valid_loss - epoch_reg_valid)
                push!(f.valid_regularizations, epoch_reg_valid)
            end

            # check if model has improved
            if epoch_train_loss < f.loss_opt
                f.loss_opt = epoch_train_loss
                f.p_opt = copy(f.p)
                f.not_improved = 0
                improvement_string = "model_best"
            else
                improvement_string = ""
                f.not_improved += 1
            end

            # track time horizon
            push!(f.time_horizons, f.time_horizon)

            # compute nominal statistic before applying the modifier
            _p_ode, _p_link, p_mod, _, p_ode, p_link = extract_params(f)

            push!(f.mod_stats_pre, debug(f.mod, _p_ode, _p_link, p_mod))
            push!(f.mod_stats_post, debug(f.mod, p_ode, p_link, p_mod))

            # clear current line then display epoch progress with mod data
            print("\033[2K")
            @printf "Epoch %d, time: %0.2f s., train loss: %0.2e, valid loss: %0.2e, lr: %0.2e, Mod: %s, stat: %0.2e %s \n" f.epoch epoch_time epoch_train_loss epoch_valid_loss f.opt[1][1].eta f.mod.name f.mod_stats_post[end] improvement_string

            # log progress every few epochs, or if training is potentially about to end
            if (f.epoch % f.logevery == 0 && f.epoch > 0) || f.not_improved >= f.patience
                # generate plots
                logscale_metrics = Dict(
                    "Train Loss" => f.epoch_train_losses,
                    "Validation Loss" => f.valid_losses,
                    "Learning Rate" => f.learn_rates,
                    "Train Regularization" => f.epoch_train_regularizations,
                    "Validation Regularization" => f.valid_regularizations,
                )
                normal_metrics = Dict(
                    "Epoch time" => f.epoch_times,
                    "Time Horizon" => f.time_horizons,
                    "Modifier Stat (nominal)" => f.mod_stats_pre,
                    "Modifier Stat (constrained)" => f.mod_stats_post,
                )
                subplot_metrics(;
                    logscale_plots = logscale_metrics,
                    normal_plots = normal_metrics,
                    savepath = f.savepath,
                )

                # save predicted vs. true trajectories plots in log
                plot_samples(f; train_set = true, savepath = f.savepath)

                # write optimal parameters and logged training metrics
                write_logs(f)

                # run additional specified callback tasks
                for task in f.callback_tasks
                    task(f)
                end
            end

            # check if training loss hasn't decreased in a while
            if f.not_improved >= f.patience
                # try increasing the time horizon
                if f.time_horizon < f.time_horizon_max
                    f.time_horizon += f.time_horizon_increment
                    @printf "Time horizon increased to %d! \n" f.time_horizon
                else
                    Flux.stop()
                end
            end

            # update epoch count
            f.epoch += 1

            # update epoch timer
            f.epoch_time = time()
        end

        # update iteration timer
        f.iteration_time = time()
    end
    return false
end

""" Plot metrics on the left and right axes """
function plot_metrics(;
    left_metrics = Dict(),
    right_metrics = Dict(),
    ylabel_left = "",
    ylabel_right = "",
    xlabel = "Epochs",
    leftlog = true,
    rightlog = false,
    savepath = "",
    filename = "metrics.png",
)
    # set up plot
    metric_plot = plot(
        size = (1000, 550),
        margin = 5 * Plots.mm,
        left_margin = right_margin = 23 * Plots.mm,
        right_margin = 23 * Plots.mm,
        label = "Training Loss",
        xlabel = xlabel,
    )

    for key in keys(left_metrics)
        plot!(
            left_metrics[key],
            yaxis = leftlog ? :log10 : :none,
            linewidth = 4,
            label = key,
            ylabel = ylabel_left,
        )
    end

    for (i, key) in enumerate(keys(right_metrics))
        plot!(
            twinx(),
            right_metrics[key],
            linecolor = i + length(left_metrics),
            linewidth = 4,
            label = key,
            legend = :right,
            yaxis = rightlog ? :log10 : :none,
            ylabel = ylabel_right,
        )
    end

    # save figure
    if savepath != ""
        savefig(metric_plot, joinpath(savepath, filename))
    end
end

""" Plot metrics as subplots """
function subplot_metrics(;
    logscale_plots = Dict(),
    normal_plots = Dict(),
    xlabel = "Epochs",
    savepath = "",
    filename = "metrics.png",
)
    num_log = length(logscale_plots)
    num_normal = length(normal_plots)
    num_plots = num_log + num_normal

    lognames = vcat(sort([k for k in keys(logscale_plots)]))
    nornames = vcat(sort([k for k in keys(normal_plots)]))
    labels = reshape(vcat(lognames, nornames), 1, num_plots)
    ylabels = reshape(vcat(["Log " * k for k in lognames], nornames), 1, num_plots)

    logdata = log10.(cat([logscale_plots[k] for k in lognames]..., dims = 2))
    nordata = cat([normal_plots[k] for k in nornames]..., dims = 2)
    data = cat(logdata, nordata, dims = 2)

    hw = Int(ceil(sqrt(num_plots)))
    plot(
        data,
        thickness_scaling = 1.0 + 1.0 / hw,
        layout = (num_log + num_normal),
        size = (500 * hw, 400 * hw),
        title = labels,
        ylabel = ylabels,
        legend = false,
    )
    xlabel!(xlabel)
    if savepath != ""
        savefig(joinpath(savepath, filename))
    end
end

""" Plot sample predictions """
function plot_samples(f::DynamicsTrainer; train_set = true, savepath = "")
    u_val, y_val = train_set ? first(f.train_data_loader) : first(f.valid_data_loader)
    predicted = predict(f, u_val; full = true)

    layout = (f.output_dim, 1)

    for k = 1:size(predicted)[3]
        plotted = plot(
            f.t,
            y_val[:, :, k]',
            label = "True",
            marker = :circle,
            linewidth = 4,
            size = (1000, f.output_dim * 500),
            margin = 5 * Plots.mm,
            layout = layout,
        )

        plot!(
            f.t,
            predicted[:, :, k]',
            label = "Predicted",
            marker = :circle,
            linewidth = 4,
            layout = layout,
        )
        xlabel!("Time")
        ylabel!("Output")
        savefig(plotted, joinpath(savepath, "predictions_$(k).png"))
    end
end

""" Write metrics & parameters """
function write_logs(f::DynamicsTrainer)
    # extract parameters
    _p_ode, _p_link, p_mod, p_feed, p_ode, p_link = extract_params(f)

    # write log
    log = Dict(
        "_p_ode" => _p_ode,
        "_p_link" => _p_link,
        "p_feed" => p_feed,
        "p_mod" => p_mod,
        "p_ode" => p_ode,
        "p_link" => p_link,
        "train_losses" => f.epoch_train_losses,
        "valid_losses" => f.valid_losses,
        "learn_rates" => f.learn_rates,
        "epoch_times" => f.epoch_times,
        "mod_stats_pre" => f.mod_stats_pre,
        "mod_stats_post" => f.mod_stats_post,
        "epoch_train_regularizations" => f.epoch_train_regularizations,
        "valid_regularizations" => f.valid_regularizations,
        "time_horizons" => f.time_horizons,
    )

    serialize(joinpath(f.savepath, "log.jls"), log)
end
