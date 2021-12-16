using ProgressBars, Printf

include("src/StableDynamics.jl")


""" Generate synthetic training data """
datapath = joinpath("data", "demo")
if ~ispath(datapath)
    mkpath(datapath)
end

# specify dynamics
function f(x,u)
    x₁,x₂,x₃ = x
    u₁,u₂ = u
    f₁ = -x₁-2x₁^3+(1+x₁^2)*u₁^2
    f₂ = -x₂+x₂^2+u₂
    f₃ = -x₃+u₂
    return [f₁, f₂, f₃]
end

A = [1 -1 2; 2 3 -1]
function h(x)
    return x
end

# data parameters
t_max = 20.0
num_knots = 2
num_samples = 300
δₜ = 0.05

# generate training data
# size (data dimension, time horizon, num_samples) 
times = collect(0.0:δₜ:t_max)
udata = zeros(2, length(times), num_samples)
ydata = zeros(3, length(times), num_samples)

@printf "Generating training data..."
for i in ProgressBar(1:num_samples)
    # generate input signal
    tspan = (0.0,t_max)
    knots = vcat(0.0, sort(t_max*rand(num_knots-2)), t_max)

    peaks = zeros(2, num_knots)
    for j=1:num_knots
        peaks[:,j] = 0.01*randn(2)
        if j>1
            peaks[:,j] += peaks[:,j-1]
        end
    end

    u = (t) -> linear_interpolate(t, knots, peaks)

    # set initial condition to be equilibrium point
    ssprob = SteadyStateProblem((x,p,t) -> f(x, u(0.0)), zeros(3))
    x₀ = Array(solve(ssprob, SSRootfind()))

    # solve controlled ODE
    prob = ODEProblem((x,p,t) -> f(x, u(t)), x₀, tspan)
    x = Array(solve(prob, saveat=times, reltol=1e-6, absol=1e-9))
    y = h(x)

    for (j,s) in enumerate(times)
        udata[:,j,i] = u(s)
    end
    ydata[:,:,i] = y
end

# dilate time
t_scale = 20.0
times *= t_scale

dataset = Dict([
    ("t", times),
    ("u", udata),
    ("y", ydata),
])

metadata = Dict([
    ("dt", δₜ),
    ("u_scale", ones(2)),
    ("u_bias", zeros(2)),
    ("y_scale", ones(3)),
    ("y_bias", zeros(3)),
    ("t_scale_factor", 1.0/t_scale),
])

serialize("data/demo/demo_data.jls", dataset)
serialize("data/demo/demo_metadata.jls", metadata)


""" Learn model """#

include("configs/config_demo.jl")
run_config(demo_config)
