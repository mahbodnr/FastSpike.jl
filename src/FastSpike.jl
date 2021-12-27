module FastSpike

export  add_group!, connect!, run!, train!, pad1D, pad2D, record!, save, load,
        plotNetwork, networkView, randomConnection, regularActivity, randomActivity, makeInput
export Network, LearningRule, NeuronType, NeuronGroup, LIF, STDP, Monitor, Connection

include("utils.jl")
include("connection.jl")
include("neuron.jl")
include("learning_rules.jl")
include("network.jl")
include("monitor.jl")
include("visualization.jl")
include("learning.jl")

end
