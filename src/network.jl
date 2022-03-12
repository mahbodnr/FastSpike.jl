export Network

mutable struct Network{T<:NeuronType}
        neurons::T
        batch_size::Int
        learning_rule::Union{LearningRule,Nothing}
        weight::AbstractArray
        adjacency::AbstractArray
        spikes::AbstractArray
        voltage::AbstractArray
        recovery::AbstractArray # membrane recovery variable, only applicable to Izhikevich neurons 
        refractory::AbstractArray
        e₊::Union{AbstractArray,Nothing}
        e₋::Union{AbstractArray,Nothing}
        learning::Bool
        groups::Dict{String,NeuronGroup}
end

function Network(
        neurons::NeuronType,
        batch_size::Int
)
        return Network(
                neurons,
                batch_size,
                nothing,
                Array{Float64}(undef, (0, 0)),
                Array{Float64}(undef, (0, 0)),
                zeros(Bool, batch_size, 0),
                ones(Float64, batch_size, 0),
                ones(Float64, batch_size, 0),
                zeros(Int64, batch_size, 0),
                nothing,
                nothing,
                false,
                Dict{String,NeuronGroup}(),
        )
end

function Network(
        neurons::NeuronType,
        batch_size::Int,
        learning_rule::Union{LearningRule,Nothing}
)
        if learning_rule.τ₊ == learning_rule.τ₋ && learning_rule.A₊ == learning_rule.A₋
                return Network(
                        neurons,
                        batch_size,
                        learning_rule,
                        Array{Float64}(undef, (0, 0)),
                        Array{Float64}(undef, (0, 0)),
                        zeros(Bool, batch_size, 0),
                        ones(Float64, batch_size, 0),
                        ones(Float64, batch_size, 0),
                        zeros(Int64, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        nothing,
                        true,
                        Dict{String,NeuronGroup}(),
                )
        else
                return Network(
                        neurons,
                        batch_size,
                        learning_rule,
                        Array{Float64}(undef, (0, 0)),
                        Array{Float64}(undef, (0, 0)),
                        zeros(Bool, batch_size, 0),
                        ones(Float64, batch_size, 0),
                        ones(Float64, batch_size, 0),
                        zeros(Int64, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        zeros(Bool, batch_size, 0),
                        true,
                        Dict{String,NeuronGroup}(),
                )
        end
end

function add_group!(network::Network, N::Int; name::Union{String,Nothing} = nothing)
        Group = NeuronGroup(N, size(network.weight, 1)+1:size(network.weight, 1)+N)

        network.weight = pad2D(network.weight, N)
        network.adjacency = pad2D(network.adjacency, N)
        if !isnothing(network.e₊)
                network.e₊ = pad1D(network.e₊, N)
        end
        if !isnothing(network.e₋)
                network.e₋ = pad1D(network.e₋, N)
        end
        network.spikes = pad1D(network.spikes, N)
        _add_neuron_features!(network, N)
        if isnothing(name)
                name = "group_$(length(network.groups)+1)"
        end
        network.groups[name] = Group

        return Group
end

function _add_neuron_features!(network::Network{LIF}, N::Int)
        network.voltage = pad1D(network.voltage, N)
        fill!(network.voltage, network.neurons.v_rest)
        network.refractory = pad1D(network.refractory, N)
end

function _add_neuron_features!(network::Network{Izhikevich}, N::Int)
        network.voltage = pad1D(network.voltage, N)
        fill!(network.voltage, network.neurons.c)
        network.recovery = network.neurons.b .* network.voltage
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
        input_voltage::Union{AbstractMatrix,Nothing} = nothing
)
        _update!(network, input_spikes, input_voltage)
        # Learning process
        if network.learning # Apply the learning rule and update weights
                train!(network, network.learning_rule;)
        end
        return network.spikes, network.voltage
end


function _update!(
        network::Network{LIF},
        input_spikes::Union{AbstractMatrix{Bool},Nothing},
        input_voltage::Union{AbstractMatrix,Nothing}
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
        network.spikes = network.voltage .>= network.neurons.v_thresh
        # External spikes
        if !isnothing(input_spikes)
                network.spikes = network.spikes .| input_spikes
        end
        # Update refractory timepoints
        network.refractory .-= network.neurons.dt
        network.refractory[network.spikes] .= network.neurons.refractory_period
end

function _update!(
        network::Network{Izhikevich},
        input_spikes::Union{AbstractMatrix{Bool},Nothing},
        input_voltage::Union{AbstractMatrix,Nothing}
)
        # Evoke spikes
        network.spikes = network.voltage .>= network.neurons.v_thresh
        # External spikes
        if !isnothing(input_spikes)
                network.spikes = network.spikes .| input_spikes
        end
        current = network.spikes * network.weight
        if !isnothing(input_voltage)
                current += input_voltage
        end
        # update voltages
        network.voltage[network.spikes] .= network.neurons.c  # change the voltage of spiked neurons to c
        network.recovery[network.spikes] .+= network.neurons.d  # add d to the recovery parameter of spiked neurons
        network.voltage += 0.04 .* network.voltage .^ 2 + 5 .* network.voltage .+ 140 - network.recovery + current
        network.recovery += network.neurons.a .* (network.neurons.b .* network.voltage - network.recovery)
end


function reset!(network::Network)
        fill!(network.spikes, 0)
        fill!(network.voltage, network.neurons.v_rest)
        if typeof(network.neurons) <: Izhikevich
                network.recovery = network.neurons.b .* network.voltage
        end
        fill!(network.refractory, 0)
        if !isnothing(network.e₊)
                fill!(network.e₊, 0)
        end
        if !isnothing(network.e₋)
                fill!(network.e₋, 0)
        end
        return
end

function makeInput(network::Network, time::Integer, inputs::Dict{NeuronGroup}, type = Float64)
        input = zeros(type, time, size(network.weight)[1])
        for group_input in inputs
                input[:, group_input[1].idx] .= group_input[2]
        end
        return input
end

function save(network::Network, filename::AbstractString)
        save_object(filename, network |> cpu)
end

function Base.getindex(network::Network, idx::Union{UnitRange{Int},Vector{Int}})
        new_e₊ = network.e₊
        new_e₋ = network.e₋
        if !isnothing(new_e₊)
                new_e₊ = new_e₊[:, idx]
        end
        if !isnothing(new_e₋)
                new_e₋ = new_e₋[:, idx]
        end
        return Network(
                network.neurons,
                network.batch_size,
                network.learning_rule,
                network.weight[idx, idx],
                network.adjacency[idx, idx],
                network.spikes[:, idx],
                network.voltage[:, idx],
                network.recovery[:, idx],
                network.refractory[:, idx],
                new_e₊,
                new_e₋,
                network.learning,
                network.groups,
        )
end