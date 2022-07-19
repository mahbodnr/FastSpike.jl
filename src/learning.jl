function train!(network::SpikingNetwork)
    return train!(network, network.learning_rule)
end

function train!(network::SpikingNetwork, learning_rule::STDP)
    if learning_rule.τ₊ == learning_rule.τ₋ && learning_rule.A₊ == learning_rule.A₋
        SymmetricalSTDP!(network, learning_rule)
    else
        AsymmetricalSTDP!(network, learning_rule)
    end
    s₊ = reshape(network.spikes, network.batch_size, 1, :)
    e₊ = reshape(learning_rule.e₊, network.batch_size, :, 1)
    s₋ = reshape(network.spikes, network.batch_size, :, 1)
    e₋ = reshape(learning_rule.e₋, network.batch_size, 1, :)
    weight_update = fill!(similar(network.weight), 0)
    # Pre-Post activities
    weight_update += ein"bix,bxj->ij"(e₊, s₊)
    # Post-Pre activities
    weight_update -= ein"bix,bxj->ij"(s₋, e₋)
    # Update weights
    learning_rule.update_rule(network, weight_update)
    network.weight = clamp.(network.weight, learning_rule.min_weight, learning_rule.max_weight)
end


function SymmetricalSTDP!(network::SpikingNetwork, learning_rule::STDP)
    learning_rule.e₊ .*= exp(-network.neurons.dt / learning_rule.τ₊)
    if learning_rule.traces_additive
        learning_rule.e₊ += learning_rule.A₊ * network.spikes
    else
        learning_rule.e₊[network.spikes] .= learning_rule.A₊
    end
    learning_rule.e₋ = learning_rule.e₊
    return
end

function AsymmetricalSTDP!(network::SpikingNetwork, learning_rule::STDP)
    learning_rule.e₊ .*= exp(-network.neurons.dt / learning_rule.τ₊)
    learning_rule.e₋ .*= exp(-network.neurons.dt / learning_rule.τ₋)
    if learning_rule.traces_additive
        learning_rule.e₊ += learning_rule.A₊ * network.spikes
        learning_rule.e₋ += learning_rule.A₋ * network.spikes
    else
        learning_rule.e₊[network.spikes] .= learning_rule.A₊
        learning_rule.e₋[network.spikes] .= learning_rule.A₋
    end
    return
end

function train!(network::SpikingNetwork, learning_rule::vSTDP)
    learning_rule.ū₊ .+= network.neurons.dt / learning_rule.τ₊ .* (-learning_rule.ū₊ + network.voltage)
    learning_rule.ū₋ .+= network.neurons.dt / learning_rule.τ₋ .* (-learning_rule.ū₋ + network.voltage)
    learning_rule.x̄ .+= network.neurons.dt / learning_rule.τₓ .* (-learning_rule.x̄ + network.spikes)

    pre_synaptic_spikes = reshape(network.spikes, network.batch_size, :, 1)
    ū₋ = reshape(learning_rule.ū₋, network.batch_size, 1, :)
    ū₊ = reshape(learning_rule.ū₊, network.batch_size, 1, :)
    u = reshape(network.voltage, network.batch_size, 1, :)
    x̄ = reshape(learning_rule.x̄, network.batch_size, :, 1)

    network.weight += (
        # LTD:
        -learning_rule.A₋ .*
        ein"bix,bxj->ij"(pre_synaptic_spikes, max.(ū₋ .- learning_rule.θ₋, 0)) +
        # LTP:
        learning_rule.A₊ .*
        ein"bix,bxj->ij"(x̄, (max.(u .- learning_rule.θ₊, 0) .* max.(ū₊ .- learning_rule.θ₋, 0)))
    )

    network.weight = clamp.(network.weight, learning_rule.min_weight, learning_rule.max_weight)
end