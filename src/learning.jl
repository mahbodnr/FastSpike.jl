function train!(network::Network, learning_rule::STDP;
    min_weight = -Inf, max_weight = Inf, softbound = false
)
    if softbound
        if abs(min_weight) == Inf || abs(max_weight) == Inf
            error("min_weight and max_weight cannot be Inf")
        end
        softbound_decay = -1 .* (network.weight .- min_weight) .* (network.weight .- max_weight)
    else
        softbound_decay = 1.0
    end
    ApplyLearningRule!(network, learning_rule, softbound_decay)
    clamp!(network.weight, min_weight, max_weight)
end

function ApplyLearningRule!(network::Network, learning_rule::STDP, softbound_decay::Union{AbstractFloat,AbstractMatrix})
    if learning_rule.τ₊ == learning_rule.τ₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
    s₊ = reshape(network.spikes, network.batch_size, 1, :)
    e₊ = reshape(network.e₊, network.batch_size, :, 1)
    s₋ = reshape(network.spikes, network.batch_size, :, 1)
    e₋ = reshape(network.e₊, network.batch_size, 1, :)
    # Pre-Post activities
    @einsum weight_update[i, j] := e₊[batch, i, x] * s₊[batch, x, j] # *w[i, j]
    # Post-Pre activities
    @einsum weight_update[i, j] -= s₋[batch, i, x] * e₋[batch, x, j] # *w[i, j]
    network.weight += weight_update .* network.adjacency .* softbound_decay
    return
end


function SymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    if learning_rule.traces_additive
        network.e₊ += learning_rule.trace_scale * network.spikes
    else
        network.e₊[network.spikes] .= learning_rule.trace_scale
    end
    network.e₋ = network.e₊
    return
end

function AsymmetricalSTDP!(network::Network, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    network.e₋ *= exp(-network.neurons.dt / learning_rule.τ₋)
    if learning_rule.traces_additive
        network.e₊ += learning_rule.A₊ * learning_rule.trace_scale * network.spikes
        network.e₋ += learning_rule.A₋ * learning_rule.trace_scale * network.spikes
    else
        network.e₊[network.spikes] .= learning_rule.A₊ * learning_rule.trace_scale
        network.e₋[network.spikes] .= learning_rule.A₋ * learning_rule.trace_scale
    end
    return
end
