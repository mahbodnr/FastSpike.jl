export DelayNetwork

"""
# DelayNetwork
Network with axonal conduction delays.
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
- `delay::AbstractArray`: Axonal conduction delays
- `_delay::AbstractArray`: Auxiliary delay array for calculations
- `delayed_voltages::AbstractArray`: Keeps track of upcomig voltages
...
"""
@with_kw mutable struct DelayNetwork{T<:NeuronType} <: SpikingNetwork{T}
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
    delay::AbstractArray = Array{Float64}(undef, (0, 0))
    _delay::AbstractArray = Array{Float64}(undef, (0, 0))
    delayed_voltages::AbstractArray = Array{Float64}(undef, (0, 0))
end

DelayNetwork(neurons) = DelayNetwork(neurons=neurons)


function add_group!(network::DelayNetwork, N::Int; name::Union{String,Nothing}=nothing)
    network.delay = pad2D(network.delay, N)
    invoke(add_group!, Tuple{SpikingNetwork,Int}, network, N; name=name)
end


"""
Add a connection between two neuron groups in the network
"""
function connect!(
    network::DelayNetwork,
    source::NeuronGroup,
    target::NeuronGroup,
    weight::AbstractArray;
    adjacency::AbstractArray=ones(Bool, source.n, target.n),
    delay::AbstractArray=zeros(Bool, source.n, target.n)
)
    network.delay[source.idx, target.idx] = delay
    network._delay = zeros(size(network.delay)..., Int(maximum(network.delay) + 1))
    for i in 1:size(delay)[1]
        for j in 1:size(delay)[2]
            network._delay[i, j, Int(network.delay[i, j] + 1)] = 1
        end
    end
    network.delayed_voltages = zeros(size(network.delay)[1], Int(maximum(network.delay) + 1))
    invoke(
        connect!,
        Tuple{SpikingNetwork,NeuronGroup,NeuronGroup,AbstractArray},
        network, source, target, weight, ; adjacency=adjacency
    )
    return
end

function connect!(
    network::DelayNetwork,
    source::NeuronGroup,
    target::NeuronGroup,
    connection::Connection;
    delay::AbstractArray=zeros(Bool, source.n, target.n)
)
    connect!(network, source, target, connection.weight;
        adjacency=connection.adjacency, delay=delay)
end

function run!(
    network::DelayNetwork;
    input_spikes::Union{AbstractMatrix{Bool},Nothing}=nothing,
    input_voltage::Union{AbstractMatrix,Nothing}=nothing
)
    if network.batch_size > 1 #TODO: implement batch_size>1 for DelayNetwork
        error("`batch_size > 1` is not supported for DelayNetwork yet!")
    end
    # Evoke spikes
    network.spikes = network.voltage .>= network.neurons.v_thresh
    # External spikes
    if !isnothing(input_spikes)
        network.spikes = network.spikes .| input_spikes
    end
    # Supress neurons #TODO     
    _update!(network, input_voltage)
    # update delayed voltages
    network.delayed_voltages[:, 1:end-1] = network.delayed_voltages[:, 2:end]
    network.delayed_voltages[:, end] .= 0
    # Learning process
    if network.learning # Apply the learning rule and update weights
        train!(network)
    end
    return network.spikes, network.voltage
end


function _update!(
    network::DelayNetwork{LIF},
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
    # update delayed voltages
    network.delayed_voltages += sum(
        network.spikes .* network.weight .* network._delay,
        dims=1
    )[1, :, :]
    current += transpose(network.delayed_voltages[:, 1])
    # update voltages
    network.voltage += current
    network.voltage[network.refractory.>0] .= network.neurons.v_rest # reset the voltage of the neurons in the refractory period
    network.voltage[network.spikes] .= network.neurons.v_reset  # change the voltage of spiked neurons to v_reset
    # Update refractory timepoints
    network.refractory .-= network.neurons.dt
    network.refractory[network.spikes] .= network.neurons.refractory_period
end

function _update!(
    network::DelayNetwork{Izhikevich},
    input_voltage::Union{AbstractMatrix,Nothing}
)

    # Get current
    current = network.spikes * network.weight
    # External voltage:
    if !isnothing(input_voltage)
        current += input_voltage
    end
    # update delayed voltages and current
    network.delayed_voltages += sum(
        transpose(network.spikes) .* network.weight .* network._delay,
        dims=1
    )[1, :, :]
    current += transpose(network.delayed_voltages[:, 1])
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
