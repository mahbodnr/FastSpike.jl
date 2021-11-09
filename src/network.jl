using CUDA
CUDA.allowscalar(false)
import ..FastSpike: LearningRule, NeuronType, pad1D, pad2D

mutable struct Network
        neurons::NeuronType
        learning_rule::LearningRule
        batch_size::UInt
        weight::CuArray
        adjacency::CuArray
        spikes::CuArray
        voltage::CuArray
        refractory::CuArray
        e₊::CuArray
        e₋::CuArray
        learning::Bool

        function Network(
                neurons::NeuronType,
                learning_rule::LearningRule,
                batch_size::Int,
                )
                if learning_rule.τ₊ == learning_rule.τ₋
                        new(
                        neurons,
                        learning_rule,
                        batch_size,
                        CuArray{Float64}(undef,(0,0)),
                        CuArray{Float64}(undef,(0,0)),
                        CUDA.zeros(Bool,batch_size,0),
                        CUDA.ones(Float64,batch_size,0),
                        CUDA.zeros(Int64,batch_size,0),
                        CuArray{Float64}(undef,(0,0)),
                        nothing,
                        true,
                        )
                else
                        new(
                        neurons,
                        learning_rule,
                        batch_size,
                        CuArray{Float64}(undef,(0,0)),
                        CuArray{Float64}(undef,(0,0)),
                        CUDA.zeros(Bool,batch_size,0),
                        CUDA.ones(Float64,batch_size,0),
                        CUDA.zeros(Int64,batch_size,0),
                        CuArray{Float64}(undef,(0,0)),
                        CuArray{Float64}(undef,(0,0)),
                        true,
                        )
                end
        end

        function Network(
                neurons::NeuronType,
                batch_size::Int,
                )
                new(
                neurons,
                nothing,
                batch_size,
                CuArray{Float64}(undef,(0,0)),
                CuArray{Float64}(undef,(0,0)),
                CUDA.zeros(Bool,batch_size,0),
                CUDA.ones(Float64,batch_size,0),
                CUDA.zeros(Int64,batch_size,0),
                nothing,
                nothing,
                false,
                )
        end
end


function add_group!(network::Network, N::Int)
        Group = NeuronGroup(N, size(network.weight,1)+1:size(network.weight,1)+N)

        network.weight = pad2D(network.weight, N)
        network.adjacency = pad2D(network.adjacency, N)
        network.eligibility = pad2D(network.eligibility, N)

        network.spikes = pad1D(network.spikes, N)
        network.voltage = pad1D(network.voltage, N)
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
        weight::CuArray,
        adjacency::CuArray,
        )
        network.weight[source.idx, target.idx] = weight
        network.adjacency[source.idx, target.idx] = adjacency
end

function connect!(
        network::Network,
        source::NeuronGroup,
        target::NeuronGroup,
        weight::CuArray,
        )
        network.weight[source.idx, target.idx] = weight
        network.adjacency[source.idx, target.idx] = CUDA.ones(Bool, source.n, target.n)
end


function run!(network::Network, input_spikes::CuArray{Bool,2}, input_voltage::CuArray)
        # Decay voltages.
        network.voltage = (
        network.neurons.voltage_decay_factor .*
        (network.voltage .- network.neurons.v_rest)
        .+ network.neurons.v_rest
        )
        # External voltage:
        input_voltage[network.refractory .> 0] = 0.0
        network.voltage += input_voltage
        # Evoke spikes
        network.spikes = network.voltage .>= network.neurons.v_thresh
        # External spikes
        network.spikes = network.spikes .| input_spikes
        # update voltages
        network.voltage += network.spikes * network.weight  # + network.bias
        network.voltage[network.refractory .> 0] = network.neurons.v_rest # reset the voltage of the neurons in the refractory period
        network.voltage[network.spikes] = network.neurons.v_reset  # change the voltage of spiked neurons to v_reset
        # Update refractory timepoints
        network.refractory .-= network.neurons.dt
        network.refractory[network.spikes]= network.neurons.refractory_period
        # Learning process
        if network.learning # Apply the learning rule and update weights
            train!(network, network.learning_rule)
        end
end



function reset(network::Network)
        fill!(network.spikes, 0)
        fill!(network.voltage, 0)
        fill!(network.eligibility, 0)
        fill!(network.refractory, 0)
end



function train!(network::Network, learning_rule::STDP)
    if learning_rule.τ₊ == learning_rule.τ₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
end

function SymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt/learning_rule.τ₊)
    if network.neurons.traces_additive
        network.e₊ += network.neurons.trace_scale * network.spikes
    else
        network.e₊[network.spikes]= network.neurons.trace_scale
    end
end

function AsymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt/learning_rule.τ₊)
    network.e₋ *= exp(-network.neurons.dt/learning_rule.τ₋)
    if network.neurons.traces_additive
        network.e₊ += network.neurons.trace_scale * network.spikes
        network.e₋ += network.neurons.trace_scale * network.spikes
    else
        network.e₊[network.spikes]= network.neurons.trace_scale
        network.e₋[network.spikes]= network.neurons.trace_scale
    end
end
