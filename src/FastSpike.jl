module FastSpike

export  add_group!, connect!, run!, train!, pad1D, pad2D, record!, save, load, gpu,
        randomConnection, regularActivity, randomActivity, makeInput
export Network, LearningRule, NeuronType, NeuronGroup, LIF, STDP, Monitor, Connection

using JLD2
function load(filename::AbstractString)
    return load_object(filename)
end

include("utils.jl")
include("connection.jl")
include("neuron.jl")
include("learning_rules.jl")
include("network.jl")
include("monitor.jl")
include("learning.jl")

end
