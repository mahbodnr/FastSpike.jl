using ..FastSpike: NeuronType, LearningRule, pad1D, pad2D, Connection
using JLD2

export Network
mutable struct Network
        neurons::NeuronType
        batch_size::Int
        learning_rule::Union{LearningRule,Nothing}
        weight::AbstractArray
        adjacency::AbstractArray
        spikes::AbstractArray
        voltage::AbstractArray
        refractory::AbstractArray
        e₊::Union{AbstractArray,Nothing}
        e₋::Union{AbstractArray,Nothing}
        learning::Bool
end

function Network(
        neurons::NeuronType,
        batch_size::Int
)
        return Network(
                neurons,
                batch_size,
                nothing,
                Array{AbstractFloat}(undef, (0, 0)),
                Array{AbstractFloat}(undef, (0, 0)),
                zeros(Bool, batch_size, 0),
                ones(AbstractFloat, batch_size, 0),
                zeros(Int64, batch_size, 0),
                nothing,
                nothing,
                false,
        )
end

function Network(
        neurons::NeuronType,
        batch_size::Int,
        learning_rule::Union{LearningRule,Nothing}
)
        if learning_rule.τ₊ == learning_rule.τ₋
                return Network(
                        neurons,
                        batch_size,
                        learning_rule,
                        Array{AbstractFloat}(undef, (0, 0)),
                        Array{AbstractFloat}(undef, (0, 0)),
                        zeros(Bool, batch_size, 0),
                        ones(AbstractFloat, batch_size, 0),
                        zeros(Int64, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        nothing,
                        true,
                )
        else
                return Network(
                        neurons,
                        batch_size,
                        learning_rule,
                        Array{AbstractFloat}(undef, (0, 0)),
                        Array{AbstractFloat}(undef, (0, 0)),
                        zeros(Bool, batch_size, 0),
                        ones(AbstractFloat, batch_size, 0),
                        zeros(Int64, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        true,
                )
        end
end

function add_group!(network::Network, N::Int)
        Group = NeuronGroup(N, size(network.weight, 1)+1:size(network.weight, 1)+N)

        network.weight = pad2D(network.weight, N)
        network.adjacency = pad2D(network.adjacency, N)
        if network.e₊ !== nothing
                network.e₊ = pad1D(network.e₊, N)
        end
        if network.e₋ !== nothing
                network.e₋ = pad1D(network.e₋, N)
        end
        network.spikes = pad1D(network.spikes, N)
        network.voltage = pad1D(network.voltage, N)
        fill!(network.voltage, network.neurons.v_rest)
        network.refractory = pad1D(network.refractory, N)

        return Group
end


"""
Add a connection between two neuron groups in the network
"""
function connect!(
        network::Network,
        source::NeuronGroup,
        target::NeuronGroup,
        weight::AbstractArray,
        adjacency::AbstractArray,
)
        network.weight[source.idx, target.idx] = weight
        network.adjacency[source.idx, target.idx] = adjacency
        return
end

function connect!(
        network::Network,
        source::NeuronGroup,
        target::NeuronGroup,
        weight::AbstractArray,
)
        network.weight[source.idx, target.idx] = weight
        network.adjacency[source.idx, target.idx] = ones(Bool, source.n, target.n)
        return
end

function connect!(
        network::Network,
        source::NeuronGroup,
        target::NeuronGroup, 
        connection::Connection,   
)
        connect!(network, source, target, connection.weight, connection.adjacency)
end

function run!(
        network::Network;
        input_spikes::Union{AbstractMatrix{Bool},Nothing} = nothing,
        input_voltage::Union{AbstractMatrix,Nothing} = nothing,
        min_weight = -Inf, max_weight = Inf, softbound = false
)
        # Decay voltages.
        network.voltage = (
                network.neurons.voltage_decay_factor .*
                (network.voltage .- network.neurons.v_rest)
                .+
                network.neurons.v_rest
        )
        # External voltage:
        if !isnothing(input_voltage)
                input_voltage[network.refractory.>0] .= 0.0
                network.voltage += input_voltage
        end
        # update voltages
        network.voltage += network.spikes * network.weight  # + network.bias
        network.voltage[network.refractory.>0] .= network.neurons.v_rest # reset the voltage of the neurons in the refractory period
        network.voltage[network.spikes] .= network.neurons.v_reset  # change the voltage of spiked neurons to v_reset
        # Evoke spikes
        if !isnothing(input_spikes)
                network.spikes = network.voltage .>= network.neurons.v_thresh
        end
        # External spikes
        network.spikes = network.spikes .| input_spikes
        # Update refractory timepoints
        network.refractory .-= network.neurons.dt
        network.refractory[network.spikes] .= network.neurons.refractory_period
        # Learning process
        if network.learning # Apply the learning rule and update weights
                train!(network, network.learning_rule;
                        min_weight = min_weight, max_weight = max_weight, softbound = softbound)
        end
        return network.spikes, network.voltage
end



function reset(network::Network)
        fill!(network.spikes, 0)
        fill!(network.voltage, 0)
        fill!(network.refractory, 0)
        if network.e₊ !== nothing
                fill!(network.e₊, 0)
        end
        if network.e₋ !== nothing
                fill!(network.e₋, 0)
        end
        return
end

function makeInput(network::Network, time::Integer, inputs::Dict{NeuronGroup})
        input = zeros(Bool, time, size(network.weight)[1])
        for group_input in inputs
                input[:, group_input[1].idx] .= group_input[2]
        end
        return input
end

function save(network::Network, filename::AbstractString)
        save_object(filename, network)
end
