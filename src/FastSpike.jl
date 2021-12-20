module FastSpike

export add_group!, connect!, run!, train!, pad1D, pad2D, record!, plotNetwork, networkView
export Network, LearningRule, NeuronType, NeuronGroup, LIF, STDP, Monitor

include("learning_rules.jl")
include("neuron.jl")
include("utils.jl")
include("network.jl")
include("learning.jl")
include("monitor.jl")
include("visualization.jl")

end
