""" StableDynamics

    Dynamics models of the form
        dxdt = f(x,u)
           y = h(x) + g(u)
    
    * dynamics: the function f
    * links: the function h
    * feedthrough: the function g
    * modifiers: a parametrized function that takes parameters of f and of h,
      and returns modified parameters of f and h. For example, implements 
      stability constraints.
    
    Catalog:

        dynamics
            affine    
                AffineOde
                LinearOde
            ctrnn0
                CTRNN0Elman
                CTRNN0Canonical
                CTRNN0Zero
            ctrnn1
                CTRNN1Nominal
                CTRNN1Canonical
                CTRNN1Zero
        links
            affine_canonical
                AffineCanonical
                LinearCanonical
        feedthrough
            linear_feedthrough
                LinearFeedthrough
                ResistiveFeedthrough
        modifiers
            Nominal
            ISSCTRNN0
            ISSCTRNN1

            regularization
                no_regularization
                lds_condition_V
                lds_condition_ctrnn0
                lds_condition_ctrnn1
"""

import DiffEqFlux: paramlength, initial_params
using DifferentialEquations, DiffEqFlux, DiffEqSensitivity, SteadyStateDiffEq, Flux, Zygote, Random
using IterTools: ncycle
using ParameterSchedulers
using ParameterSchedulers: Stateful, next!
using Printf

include("dynamics/dynamics.jl")
include("links/links.jl")
include("modifiers/modifiers.jl")
include("modifiers/regularization.jl")
include("feedthrough/feedthrough.jl")
include("interpolator.jl")
include("trainer.jl")
include("forward_solve.jl")
include("initial_condition.jl")
include("configurations.jl")
