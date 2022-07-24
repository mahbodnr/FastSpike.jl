function train!(network::SpikingNetwork)
    return train!(network, network.learning_rule)
    network.weight = clamp.(network.weight, network.learning_rule.min_weight, network / learning_rule.max_weight)
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
    ) .* network.adjacency
end

function train!(network::SpikingNetwork, learning_rule::cSTDP)
    if isnothing(learning_rule.initial_weights)
        learning_rule.initial_weights = copy(network.weight)
        learning_rule.efficacy[randsubseq(1:length(learning_rule.efficacy), 1 - learning_rule.β)] .= 1
        learning_rule.efficacy .*= network.adjacency
    end
    learning_rule.calcium .+= network.neurons.dt / learning_rule.τ_calcium .* (-learning_rule.calcium)
    learning_rule.calcium[network.spikes[1, :], :] .+= learning_rule.Cₚᵣₑ
    learning_rule.calcium[:, network.spikes[1, :]] .+= learning_rule.Cₚₒₛₜ
    # Gaussian = similar(learning_rule.calcium)
    # CUDA.@allowscalar rand!(Normal(0, 1), Gaussian)
    learning_rule.efficacy .+= network.neurons.dt / learning_rule.τᵨ .* (
        -learning_rule.efficacy .* (1 .- learning_rule.efficacy) .* (learning_rule.ρ_star .- learning_rule.efficacy)
        +
        learning_rule.γ₊ .* (1 .- learning_rule.efficacy) .* Θ.(learning_rule.calcium .- learning_rule.θ₊)
        -
        learning_rule.γ₋ .* learning_rule.efficacy .* Θ.(learning_rule.calcium .- learning_rule.θ₋)
        # +
        # learning_rule.σ .* sqrt(learning_rule.τᵨ) .* Θ.(learning_rule.calcium .- min(learning_rule.θ₋, learning_rule.θ₊)) * Gaussian
    ) .* network.adjacency
    network.weight = learning_rule.initial_weights .* learning_rule.efficacy
end