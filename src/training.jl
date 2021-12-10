using ..FastSpike: Network, STDP
using Einsum

function train!(network::Network, learning_rule::STDP)
    if learning_rule.τ₊ == learning_rule.τ₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
    s = reshape(network.spikes, network.batch_size, :, 1)
    e₊ = reshape(network.e₊, network.batch_size, 1, :)
    e₋ = reshape(network.e₊, network.batch_size, 1, :)
    w = network.weight
    @einsum weight_update[i, j] := s[batch, i, x] * e₊[batch, x, j] * w[i, j]
    @einsum weight_update[i, j] += s[batch, i, x] * e₋[batch, x, j] * w[i, j]
    network.weight += weight_update
end

function SymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    if network.neurons.traces_additive
        network.e₊ += network.neurons.trace_scale * network.spikes
    else
        network.e₊[network.spikes] .= network.neurons.trace_scale
    end
end

function AsymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    network.e₋ *= exp(-network.neurons.dt / learning_rule.τ₋)
    if network.neurons.traces_additive
        network.e₊ += network.neurons.trace_scale * network.spikes
        network.e₋ += network.neurons.trace_scale * network.spikes
    else
        network.e₊[network.spikes] .= network.neurons.trace_scale
        network.e₋[network.spikes] .= network.neurons.trace_scale
    end
end
