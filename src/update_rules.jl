export regular_update, weight_dependent_update, softmax_update

function regular_update(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency
    return
end

function weight_dependent_update(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency .* network.weight
    return
end

function softbound_update(network::SpikingNetwork, weight_update::AbstractArray)
    # if abs(network.learning_rule.min_weight) == Inf || abs(network.learning_rule.max_weight) == Inf
    #     error("min_weight and max_weight cannot be Inf")
    # end
    network.weight += (
        weight_update .* network.adjacency .*
        (network.weight .- network.learning_rule.min_weight) .*
        (network.learning_rule.max_weight .- network.weight)
    )
    return
end
