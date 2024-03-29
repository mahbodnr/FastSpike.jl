export Network

""" 
# Network
Network structs are used to run simulations. It contains all neurons and connections.
# Arguments
- `neurons::NeuronType`
- `batch_size::Integer=1`
...
- `weight::AbstractArray`: Final size= (#neurons, #neurons)
- `adjacency::AbstractArray`: Final size= (#neurons, #neurons)
- `spikes::AbstractArray`: Final size= (Batch size, #neurons)
- `voltage::AbstractArray`: Final size= (Batch size, #neurons)
- `recovery::AbstractArray`: Membrane recovery variable, only applicable to Izhikevich neurons 
- `refractory::AbstractArray`: Refractory period, only applicable to LIF neurons. Final size= (Batch size, #neurons)
...
"""
@with_kw mutable struct Network{T<:NeuronType} <: SpikingNetwork{T}
        neurons::T
        batch_size::Int = 1
        learning_rule::Union{LearningRule,Nothing} = nothing
        weight::AbstractArray = Array{Float64}(undef, (0, 0))
        adjacency::AbstractArray = Array{Bool}(undef, (0, 0))
        spikes::AbstractArray = zeros(Bool, batch_size, 0)
        voltage::AbstractArray = ones(Float32, batch_size, 0)
        recovery::AbstractArray = ones(Float32, batch_size, 0)
        refractory::AbstractArray = zeros(Int32, batch_size, 0)
        learning::Bool = isnothing(learning_rule) ? false : true
        groups::Dict{String,NeuronGroup} = Dict{String,NeuronGroup}()
end
#TODO: add recovery and refractory to LIF and Izhikevich neurons
Network(neurons) = Network(neurons=neurons)

function add_group!(network::SpikingNetwork, N::Int; name::Union{String,Nothing}=nothing)
        Group = NeuronGroup(N, size(network.weight, 1)+1:size(network.weight, 1)+N)
        network.weight = pad2D(network.weight, N)
        network.adjacency = pad2D(network.adjacency, N)
        network.spikes = pad1D(network.spikes, N)
        add_group!(network.neurons, N, network.batch_size)
        if !isnothing(network.learning_rule)
                add_group!(network.learning_rule, N, network.batch_size)
        end
        _add_neuron_features!(network, N)
        if isnothing(name)
                name = "group_$(length(network.groups)+1)"
        end
        network.groups[name] = Group
        return Group
end

#TODO: move this function to neuron add_group!
function _add_neuron_features!(network::SpikingNetwork{LIF}, N::Int)
        network.voltage = pad1D(network.voltage, N)
        fill!(network.voltage, network.neurons.v_rest)
        network.refractory = pad1D(network.refractory, N)
end

function _add_neuron_features!(network::SpikingNetwork{Izhikevich}, N::Int)
        network.voltage = pad1D(network.voltage, N)
        if typeof(network.neurons.c) <: AbstractArray
                network.voltage = network.neurons.c[:, 1:size(network.voltage, 2)]
        else
                fill!(network.voltage, network.neurons.c)
        end
        if typeof(network.neurons.b) <: AbstractArray
                network.recovery = network.neurons.b[:, 1:size(network.voltage, 2)] .* network.voltage
        else
                network.recovery = network.neurons.b .* network.voltage
        end
end

function _add_neuron_features!(network::SpikingNetwork{AdEx}, N::Int)
        network.voltage = pad1D(network.voltage, N)
        fill!(network.voltage, network.neurons.v_reset)
end

"""
Add a connection between two neuron groups in the network
"""
function connect!(
        network::SpikingNetwork,
        source::NeuronGroup,
        target::NeuronGroup,
        weight::AbstractArray;
        adjacency::AbstractArray=ones(Bool, source.n, target.n)
)
        network.weight[source.idx, target.idx] = weight
        network.adjacency[source.idx, target.idx] = adjacency
        return
end

function connect!(
        network::SpikingNetwork,
        source::NeuronGroup,
        target::NeuronGroup,
        connection::Connection,
)
        connect!(network, source, target, connection.weight; adjacency=connection.adjacency)
end

function run!(
        network::Network;
        input_spikes::Union{AbstractMatrix{Bool},Nothing}=nothing,
        input_voltage::Union{AbstractMatrix,Nothing}=nothing
)
        # Evoke spikes
        network.spikes = network.voltage .>= network.neurons.v_thresh
        # External spikes
        if !isnothing(input_spikes)
                network.spikes = network.spikes .| input_spikes
        end
        # Supress neurons #TODO     
        _update!(network, input_voltage)
        # Learning process
        if network.learning # Apply the learning rule and update weights
                train!(network)
        end
        return network.spikes, network.voltage
end

function _update!(
        network::Network{LIF},
        input_voltage::Union{AbstractMatrix,Nothing}
)
        # Decay voltages.
        network.voltage = (
                network.neurons.voltage_decay_factor .*
                (network.voltage .- network.neurons.v_rest)
                .+
                network.neurons.v_rest
        )
        # Get current
        current = network.spikes * network.weight  # + network.bias #TODO
        # External voltage:
        if !isnothing(input_voltage)
                input_voltage[network.refractory.>0] .= 0.0
                network.voltage += input_voltage
        end
        # update voltages
        network.voltage += current
        network.voltage[network.refractory.>0] .= network.neurons.v_rest # reset the voltage of the neurons in the refractory period
        network.voltage[network.spikes] .= network.neurons.v_reset  # change the voltage of spiked neurons to v_reset
        # Update refractory timepoints
        network.refractory .-= network.neurons.dt
        network.refractory[network.spikes] .= network.neurons.refractory_period
end

function _update!(
        network::Network{AdEx},
        input_voltage::Union{AbstractMatrix,Nothing}
)
        neurons = network.neurons
        # Get current
        current = network.spikes * network.weight  # + network.bias #TODO
        # External voltage:
        if !isnothing(input_voltage)
                current += input_voltage
        end
        # update voltages
        network.voltage += neurons.dt / neurons.C .* (
                neurons.gₗ .* (
                        -network.voltage .+ neurons.Eₗ + neurons.Δₜ .* exp.((network.voltage - neurons.v_thresh) ./ neurons.Δₜ)
                ) + 1000 .* current - neurons.wₐ + neurons.z
        )
        neurons.wₐ += neurons.dt / neurons.τₐ .* (
                neurons.a * (network.voltage .- neurons.Eₗ) - neurons.wₐ
        )
        neurons.z += neurons.dt / neurons.τz .* (-neurons.z)
        neurons.v_thresh += -neurons.dt / neurons.τᵥ .* (
                neurons.v_thresh .- neurons.Vₜ_rest
        )
        network.voltage[network.spikes] .= neurons.v_reset  # change the voltage of spiked neurons to v_reset
        neurons.wₐ[network.spikes] .+= neurons.b
        neurons.z[network.spikes] .= neurons.Iₛ
        neurons.v_thresh[network.spikes] .= neurons.Vₜ_max
end

function _update!(
        network::Network{Izhikevich},
        input_voltage::Union{AbstractMatrix,Nothing}
)
        # Get current
        current = network.spikes * network.weight
        # External voltage:
        if !isnothing(input_voltage)
                current += input_voltage
        end
        # Update voltages (spiked neurons)
        network.voltage[network.spikes] .= (network.spikes.*network.neurons.c)[network.spikes]  # change the voltage of spiked neurons to c #TODO: optimize for scalar values
        network.recovery[network.spikes] .+= (network.spikes.*network.neurons.d)[network.spikes]  # add d to the recovery parameter of spiked neurons #TODO: optimize for scalar values
        # Update voltages
        for _ in 1:network.neurons.runge_kutta_order
                network.voltage += (network.neurons.dt / network.neurons.runge_kutta_order) .* (
                        0.04 .* network.voltage .^ 2 + 5 .* network.voltage .+ 140 - network.recovery + current
                )
        end
        network.recovery += network.neurons.dt .* (
                network.neurons.a .* (network.neurons.b .* network.voltage - network.recovery)
        )
end


function _reset!(network::SpikingNetwork)
        fill!(network.spikes, 0)
        if !isnothing(network.learning_rule)
                reset!(network.learning_rule)
        end
        return
end

function reset!(network::SpikingNetwork{LIF})
        fill!(network.voltage, network.neurons.v_rest)
        fill!(network.refractory, 0)
        _reset!(network)
end

function reset!(network::SpikingNetwork{Izhikevich})
        network.voltage .= network.neurons.c
        network.recovery = network.neurons.b .* network.voltage
        _reset!(network)
end

function makeInput(network::SpikingNetwork, time::Integer, inputs::Dict{NeuronGroup}, type=Float64)
        input = zeros(type, time, size(network.weight)[1])
        for group_input in inputs
                input[:, group_input[1].idx] .= group_input[2]
        end
        return input
end

function save(network::SpikingNetwork, filename::AbstractString)
        save_object(filename, network |> cpu)
end


function Base.getindex(network::SpikingNetwork, idx::Union{UnitRange{Int},Vector{Int},Int})
        typeof(network)(
                [
                        begin
                                if typeof(getfield(network, property)) <: AbstractArray
                                        if length(getfield(network, property)) > 0
                                                getfield(network, property)[idx]
                                        else
                                                getfield(network, property)
                                        end
                                else
                                        getfield(network, property)
                                end
                        end for property in propertynames(network)
                ]...
        )
end

function Base.getindex(network::SpikingNetwork, group::NeuronGroup)
        return network[group.idx]
end
