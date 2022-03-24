using ProgressBars, Printf, LinearAlgebra, Random

include("src/StableDynamics.jl")

# set random seed
Random.seed!(0);


""" Generate synthetic training data """
datapath = joinpath("data", "demo")
if ~ispath(datapath)
    mkpath(datapath)
end

n, m = 3, 2

function sample_params(n, m)
    function _sample()
        W = randn(n, n)
        A = randn(n, n)
        B = randn(n, m)
        return W, A, B
    end
    W, A, B = _sample()
    while maximum(eigvals(W*A-I(n) + A'*W'-I(n))) >= 0
        W, A, B = _sample()
    end
    return W,A,B
end

W, A, B = sample_params(n, m)

# specify dynamics
function f(x,u)
    return -x + W*tanh.(A*x + B*u)
end

# data parameters
t_max = 50.0
num_knots = 20
num_samples = 300
δₜ = 0.1

# generate training data
# size (data dimension, time horizon, num_samples) 
times = collect(0.0:δₜ:t_max)
udata = zeros(m, length(times), num_samples)
ydata = zeros(n, length(times), num_samples)

@printf "Generating training data..."
for i in ProgressBar(1:num_samples)
    # generate input signal
    tspan = (0.0, t_max)
    knots = vcat(0.0, sort(t_max*rand(num_knots-2)), t_max)

    peaks = zeros(2, num_knots)
    for j=1:num_knots
        peaks[:,j] = 2.0*(rand(2) .- 0.5)
    end

    u = (t) -> linear_interpolate(t, knots, peaks)

    # set initial condition to be equilibrium point
    ssprob = SteadyStateProblem((x,p,t) -> f(x, u(0.0)), zeros(3))
    x₀ = Array(solve(ssprob, SSRootfind()))

    # solve controlled ODE
    prob = ODEProblem((x,p,t) -> f(x, u(t)), x₀, tspan)
    y = Array(solve(prob, saveat=times, reltol=1e-6, absol=1e-9))

    for (j,s) in enumerate(times)
        udata[:,j,i] = u(s)
    end

    ydata[:,:,i] = y
end

# dilate time
t_scale = 10.0
times *= t_scale

# scale inputs and outputs
umin = minimum(udata, dims=[2,3])
umax = maximum(udata, dims=[2,3])
ymin = minimum(ydata, dims=[2,3])
ymax = maximum(ydata, dims=[2,3])

uscale = 0.5*(umax-umin)
yscale = 0.5*(ymax-ymin)

ubias = -umin./uscale .- 1
ybias = -ymin./yscale .- 1

udata = udata./uscale .+ ubias
ydata = ydata./yscale .+ ybias

dataset = Dict([
    ("t", times),
    ("u", udata),
    ("y", ydata),
])

metadata = Dict([
    ("dt", δₜ),
    ("u_scale", uscale),
    ("u_bias", ubias),
    ("y_scale", yscale),
    ("y_bias", ybias),
    ("t_scale_factor", 1.0/t_scale),
])

serialize("data/demo/demo_data.jls", dataset)
serialize("data/demo/demo_metadata.jls", metadata)


""" Learn model """#

include("configs/config_demo.jl")
run_config(demo_config)
