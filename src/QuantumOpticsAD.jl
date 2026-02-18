module QuantumOpticsAD

using Reexport
@reexport using QuantumOpticsBase
using LinearAlgebra, SciMLBase, SciMLSensitivity, OrdinaryDiffEq, NLsolve, FFTW

include("types.jl")
include("dynamics.jl")
include("sensitivity.jl")
include("steadystate.jl")
include("correlations.jl") 

export ADKet, wrap_state, unwrap_state
export schroedinger_ad, master_ad, steadystate_ad
export correlation_ad, spectrum_ad 
end
#to do
# SciML Optimization Wrappers
# Integrate KernelAbstractions.jl to support CUDA/Metal/AMD GPU arrays