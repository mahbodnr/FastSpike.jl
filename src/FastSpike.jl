module FastSpike

export add_group!, connect!, run!, pad1D, pad2D
export Network, LearningRule, NeuronType, NeuronGroup, LIF

include("learning_rules.jl")
include("neuron.jl")
include("utils.jl")
include("network.jl")

end
