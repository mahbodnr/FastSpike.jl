function train!(network::SpikingNetwork)
    if network.learning_rule.τ₊ == network.learning_rule.τ₋ && network.learning_rule.A₊ == network.learning_rule.A₋
        SymmetricalSTDP!(network)
    else
        AsymmetricalSTDP!(network)
    end

    s₊ = reshape(network.spikes, network.batch_size, 1, :)
    e₊ = reshape(network.learning_rule.e₊, network.batch_size, :, 1)
    s₋ = reshape(network.spikes, network.batch_size, :, 1)
    e₋ = reshape(network.learning_rule.e₋, network.batch_size, 1, :)

    weight_update = fill!(similar(network.weight), 0)
    # Pre-Post activities
    weight_update += ein"bix,bxj->ij"(e₊, s₊)
    # Post-Pre activities
    weight_update -= ein"bix,bxj->ij"(s₋, e₋)

    # Update weights
    network.learning_rule.update_rule(network, weight_update)

    network.weight = clamp.(network.weight, network.learning_rule.min_weight, network.learning_rule.max_weight)
end



function SymmetricalSTDP!(network::SpikingNetwork)
    network.learning_rule.e₊ .*= exp(-network.neurons.dt / network.learning_rule.τ₊)
    if network.learning_rule.traces_additive
        network.learning_rule.e₊ += network.learning_rule.A₊ * network.spikes
    else
        network.learning_rule.e₊[network.spikes] .= network.learning_rule.A₊
    end
    network.e₋ = network.learning_rule.e₊
    return
end

function AsymmetricalSTDP!(network::SpikingNetwork)
    network.learning_rule.e₊ .*= exp(-network.neurons.dt / network.learning_rule.τ₊)
    network.learning_rule.e₋ .*= exp(-network.neurons.dt / network.learning_rule.τ₋)
    if network.learning_rule.traces_additive
        network.learning_rule.e₊ += network.learning_rule.A₊ * network.spikes
        network.learning_rule.e₋ += network.learning_rule.A₋ * network.spikes
    else
        network.learning_rule.e₊[network.spikes] .= network.learning_rule.A₊
        network.learning_rule.e₋[network.spikes] .= network.learning_rule.A₋
    end
    return
end
