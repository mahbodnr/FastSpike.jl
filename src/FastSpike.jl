module FastSpike

using JLD2
using Random
using NNlib
using CUDA
using Adapt
using OMEinsum

export  add_group!, connect!, run!, train!, reset!, pad1D, pad2D, record!, save, load, gpu,
        randomConnection, regularActivity, randomActivity, makeInput, PSP, EPSP, IPSP
export Network, LearningRule, NeuronType, NeuronGroup, LIF, STDP, Monitor, Connection

include("utils.jl")
include("connection.jl")
include("neuron.jl")
include("learning_rules.jl")
include("network.jl")
include("monitor.jl")
include("learning.jl")
include("cuda.jl")

end
