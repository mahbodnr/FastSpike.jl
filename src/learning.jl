function train!(network::Union{Network,DelayNetwork}, learning_rule::STDP)
    if learning_rule.softbound
        if abs(learning_rule.min_weight) == Inf || abs(learning_rule.max_weight) == Inf
            error("min_weight and max_weight cannot be Inf")
        end
        softbound_decay = begin
            (network.weight .- learning_rule.min_weight) .*
            (learning_rule.max_weight .- network.weight)
        end
    else
        softbound_decay = 1.0
    end
    ApplyLearningRule!(network, learning_rule, softbound_decay)
    network.weight = clamp.(network.weight, learning_rule.min_weight, learning_rule.max_weight)
end

function ApplyLearningRule!(network::Network, learning_rule::STDP, softbound_decay::Union{AbstractFloat,AbstractMatrix})
    if learning_rule.τ₊ == learning_rule.τ₋ && learning_rule.A₊ == learning_rule.A₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
    s₊ = reshape(network.spikes, network.batch_size, 1, :)
    e₊ = reshape(network.e₊, network.batch_size, :, 1)
    s₋ = reshape(network.spikes, network.batch_size, :, 1)
    e₋ = reshape(network.e₋, network.batch_size, 1, :)
    weight_update = fill!(similar(network.weight), 0)
    # Pre-Post activities
    weight_update += ein"bix,bxj->ij"(e₊, s₊)
    # Post-Pre activities
    weight_update -= ein"bix,bxj->ij"(s₋, e₋)
    # Update weights
    network.weight += weight_update .* network.adjacency .* softbound_decay
    return
end

function ApplyLearningRule!(network::DelayNetwork, learning_rule::STDP, softbound_decay::Union{AbstractFloat,AbstractMatrix})
    if learning_rule.τ₊ == learning_rule.τ₋ && learning_rule.A₊ == learning_rule.A₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
    s₊ = reshape(network.spikes, 1, 1, :)
    e₊ = reshape(network.e₊, 1, :, 1)
    s₋ = reshape(network.spikes, 1, :, 1)
    e₋ = reshape(network.e₋, 1, 1, :)
    weight_update = fill!(similar(network.weight), 0)
    # Pre-Post activities
    weight_update += ein"bix,bxj->ij"(e₊, s₊)
    # Post-Pre activities
    weight_update -= ein"bix,bxj->ij"(s₋, e₋)
    # Update weights
    network.weight += weight_update .* network.adjacency .* softbound_decay
    return
end

function SymmetricalSTDP!(network::Union{Network,DelayNetwork}, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    if learning_rule.traces_additive
        network.e₊ += learning_rule.A₊ * network.spikes
    else
        network.e₊[network.spikes] .= learning_rule.A₊
    end
    network.e₋ = network.e₊
    return
end

function AsymmetricalSTDP!(network::Union{Network,DelayNetwork}, learning_rule::STDP)
    network.e₊ *= exp(-network.neurons.dt / learning_rule.τ₊)
    network.e₋ *= exp(-network.neurons.dt / learning_rule.τ₋)
    if learning_rule.traces_additive
        network.e₊ += learning_rule.A₊ * network.spikes
        network.e₋ += learning_rule.A₋ * network.spikes
    else
        network.e₊[network.spikes] .= learning_rule.A₊
        network.e₋[network.spikes] .= learning_rule.A₋
    end
    return
end
