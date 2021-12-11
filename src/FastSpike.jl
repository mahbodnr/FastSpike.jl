module FastSpike

export add_group!, connect!, run!, train!, pad1D, pad2D
export Network, LearningRule, NeuronType, NeuronGroup, LIF, STDP

include("learning_rules.jl")
include("neuron.jl")
include("utils.jl")
include("network.jl")
include("learning.jl")

end
