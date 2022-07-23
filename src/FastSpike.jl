module FastSpike

using LinearAlgebra
using Random
using Distributions
using RecursiveArrayTools
using JLD2
using NNlib
using CUDA
using Adapt
using OMEinsum
using Parameters: @with_kw

export add_group!, connect!, run!, train!, reset!, pad1D, pad2D, record!, get, save, load,
        gpu, cpu, randomConnection, regularActivity, randomActivity, makeInput, EI,
        raster, set_reward
export SpikingNetwork, Network, DelayNetwork, LearningRule, NeuronType, NeuronGroup, LIF,
        Izhikevich, AdEx, STDP, vSTDP, cSTDP, Monitor, Connection, UpdateRule, RewardModulatedUpdateRule,
        RegularUpdate, WeightDependent, WeightDependentUpdate, Softbound, RewardModulated,
        WeightDependentRewardModulated

abstract type SpikingNetwork{T} end

include("utils.jl")
include("connection.jl")
include("neuron.jl")
include("update_rules.jl")
include("learning_rules.jl")
include("network.jl")
include("delay_network.jl")
include("monitor.jl")
include("learning.jl")
include("cuda.jl")

end
