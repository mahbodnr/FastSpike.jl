module FastSpike

import Base.@kwdef
using LinearAlgebra
using Random
using RecursiveArrayTools
using JLD2
using NNlib
using CUDA
using Adapt
using OMEinsum

export add_group!, connect!, run!, train!, reset!, pad1D, pad2D, record!, get, save, load,
        gpu, cpu, randomConnection, regularActivity, randomActivity, makeInput, EI,
        raster, regular_update, weight_dependent_update, softbound_update
export Network, DelayNetwork, LearningRule, NeuronType, NeuronGroup, LIF, Izhikevich,
        STDP, Monitor, Connection

include("utils.jl")
include("connection.jl")
include("neuron.jl")
include("learning_rules.jl")
include("network.jl")
include("delay_network.jl")
include("update_rules.jl")
include("monitor.jl")
include("learning.jl")
include("cuda.jl")

end
