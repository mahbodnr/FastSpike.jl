export DelayNetwork

"""Network with axonal conduction delays."""
mutable struct DelayNetwork{T<:NeuronType}
    neurons::T
    learning_rule::Union{LearningRule,Nothing}
    weight::AbstractArray
    adjacency::AbstractArray
    delay::AbstractArray # axonal conduction delays
    _delay::AbstractArray # auxiliary delay array for calculations
    delayed_voltages::AbstractArray # keeps track of upcomig voltages
    spikes::AbstractArray
    voltage::AbstractArray
    recovery::AbstractArray # membrane recovery variable, only applicable to Izhikevich neurons 
    refractory::AbstractArray
    e₊::Union{AbstractArray,Nothing}
    e₋::Union{AbstractArray,Nothing}
    learning::Bool
    groups::Dict{String,NeuronGroup}
end

function DelayNetwork(
    neurons::NeuronType,
)
    return DelayNetwork(
        neurons,
        nothing,
        Array{Float64}(undef, (0, 0)),
        Array{Float64}(undef, (0, 0)),
        Array{Float64}(undef, (0, 0)),
        Array{Float64}(undef, (0, 0)),
        Array{Float64}(undef, (0, 0)),
        zeros(Bool, 1, 0),
        ones(Float64, 1, 0),
        ones(Float64, 1, 0),
        zeros(Int64, 1, 0),
        nothing,
        nothing,
        false,
        Dict{String,NeuronGroup}(),
    )
end

function DelayNetwork(
    neurons::NeuronType,
    learning_rule::Union{LearningRule,Nothing}
)
    if learning_rule.τ₊ == learning_rule.τ₋ && learning_rule.A₊ == learning_rule.A₋
        return DelayNetwork(
            neurons,
            learning_rule,
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            zeros(Bool, 1, 0),
            ones(Float64, 1, 0),
            ones(Float64, 1, 0),
            zeros(Int64, 1, 0),
            zeros(Bool, 1, 0),
            nothing,
            true,
            Dict{String,NeuronGroup}(),
        )
    else
        return DelayNetwork(
            neurons,
            learning_rule,
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            Array{Float64}(undef, (0, 0)),
            zeros(Bool, 1, 0),
            ones(Float64, 1, 0),
            ones(Float64, 1, 0),
            zeros(Int64, 1, 0),
            zeros(Bool, 1, 0),
            zeros(Bool, 1, 0),
            true,
            Dict{String,NeuronGroup}(),
        )
    end
end

function add_group!(network::DelayNetwork, N::Int; name::Union{String,Nothing} = nothing)
    Group = NeuronGroup(N, size(network.weight, 1)+1:size(network.weight, 1)+N)

    network.weight = pad2D(network.weight, N)
    network.adjacency = pad2D(network.adjacency, N)
    network.delay = pad2D(network.delay, N)
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

function _add_neuron_features!(network::DelayNetwork{LIF}, N::Int)
    network.voltage = pad1D(network.voltage, N)
    fill!(network.voltage, network.neurons.v_rest)
    network.refractory = pad1D(network.refractory, N)
end

function _add_neuron_features!(network::DelayNetwork{Izhikevich}, N::Int)
    network.voltage = pad1D(network.voltage, N)
    fill!(network.voltage, network.neurons.c)
    network.recovery = network.neurons.b .* network.voltage
end


"""
Add a connection between two neuron groups in the network
"""
function connect!(
    network::DelayNetwork,
    source::NeuronGroup,
    target::NeuronGroup,
    weight::AbstractArray,
    adjacency::AbstractArray,
    delay::AbstractArray,
)
    network.weight[source.idx, target.idx] = weight
    network.adjacency[source.idx, target.idx] = adjacency
    network.delay[source.idx, target.idx] = delay

    network._delay = zeros(size(network.delay)..., Int(maximum(network.delay) + 1))
    for i in 1:size(delay)[1]
        for j in 1:size(delay)[2]
            network._delay[i, j, Int(network.delay[i, j] + 1)] = 1
        end
    end
    network.delayed_voltages = zeros(size(network.delay)[1], Int(maximum(network.delay) + 1))
    return
end

function connect!(
    network::DelayNetwork,
    source::NeuronGroup,
    target::NeuronGroup,
    weight::AbstractArray,
)
    return connect!(
        network,
        source,
        target,
        weight,
        ones(Bool, source.n, target.n),
        zeros(Int64, source.n, target.n),
    )
end

function connect!(
    network::DelayNetwork,
    source::NeuronGroup,
    target::NeuronGroup,
    connection::Connection,
    delay::AbstractArray,
)
    connect!(network, source, target, connection.weight, connection.adjacency, delay)
end

function run!(
    network::DelayNetwork;
    input_spikes::Union{AbstractMatrix{Bool},Nothing} = nothing,
    input_voltage::Union{AbstractMatrix,Nothing} = nothing
)
    _update!(network, input_spikes, input_voltage)
    # update delayed voltages
    network.delayed_voltages[:, 1:end-1] = network.delayed_voltages[:, 2:end]
    network.delayed_voltages[:, end] .= 0
    # Learning process
    if network.learning # Apply the learning rule and update weights
        train!(network, network.learning_rule;)
    end
    return network.spikes, network.voltage
end


function _update!(
    network::DelayNetwork{LIF},
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
    # update delayed voltages
    network.delayed_voltages += sum(
        network.spikes .* network.weight .* network._delay,
        dims = 1
    )[1, :, :]
    # update voltages
    network.voltage += network.delayed_voltages[:, 1]
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
    network::DelayNetwork{Izhikevich},
    input_spikes::Union{AbstractMatrix{Bool},Nothing},
    input_voltage::Union{AbstractMatrix,Nothing}
)
    # Evoke spikes
    network.spikes = network.voltage .>= network.neurons.v_thresh
    # External spikes
    if !isnothing(input_spikes)
        network.spikes = network.spikes .| input_spikes
    end
    # update delayed voltages and current
    network.delayed_voltages += sum(
        transpose(network.spikes) .* network.weight .* network._delay,
        dims = 1
    )[1, :, :]
    current = transpose(network.delayed_voltages[:, 1])
    # External voltage:
    if !isnothing(input_voltage)
        current += input_voltage
    end
    # update voltages
    network.voltage[network.spikes] .= (network.spikes.*network.neurons.c)[network.spikes]  # change the voltage of spiked neurons to c #TODO: optimize for scalar values
    network.recovery[network.spikes] .+= (network.spikes.*network.neurons.d)[network.spikes]  # add d to the recovery parameter of spiked neurons #TODO: optimize for scalar values
    network.voltage += 0.04 .* network.voltage .^ 2 + 5 .* network.voltage .+ 140 - network.recovery + current
    network.recovery += network.neurons.a .* (network.neurons.b .* network.voltage - network.recovery)
end


function reset!(network::DelayNetwork)
    fill!(network.spikes, 0)
    if typeof(network.neurons) <: Izhikevich
        network.recovery = network.neurons.b .* network.voltage
    else
        fill!(network.voltage, network.neurons.v_rest)
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

function makeInput(network::DelayNetwork, time::Integer, inputs::Dict{NeuronGroup}, type = Float64)
    input = zeros(type, time, size(network.weight)[1])
    for group_input in inputs
        input[:, group_input[1].idx] .= group_input[2]
    end
    return input
end

function save(network::DelayNetwork, filename::AbstractString)
    save_object(filename, network |> cpu)
end

function Base.getindex(network::DelayNetwork, idx::Union{UnitRange{Int},Vector{Int}})
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