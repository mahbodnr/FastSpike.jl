export UpdateRule, RewardModulatedUpdateRule
export RegularUpdate, WeightDependentUpdate, SoftboundUpdate, RewardModulatedUpdate, WeightDependentRewardModulatedUpdate
export set_reward

abstract type UpdateRule end

struct RegularUpdate <: UpdateRule end
function (update_rule::RegularUpdate)(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency
end

struct WeightDependentUpdate <: UpdateRule end
function (update_rule::WeightDependentUpdate)(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency .* network.weight
end

struct SoftboundUpdate <: UpdateRule end
function (update_rule::SoftboundUpdate)(network::SpikingNetwork, weight_update::AbstractArray)
    # if abs(network.learning_rule.min_weight) == Inf || abs(network.learning_rule.max_weight) == Inf
    #     error("min_weight and max_weight cannot be Inf")
    # end
    network.weight += (
        weight_update .* network.adjacency .*
        (network.weight .- network.learning_rule.min_weight) .*
        (network.learning_rule.max_weight .- network.weight)
    )
end

abstract type RewardModulatedUpdateRule <: UpdateRule end
"""
# RSTDP learning rule
Reward-Modulated Spike Timing Dependent Plasticity learning rule.
Based on: Izhikevich, E.M. Solving the distal reward problem through linkage of STDP and dopamine signaling. BMC Neurosci 8, S15 (2007). https://doi.org/10.1186/1471-2202-8-S2-S15
# Arguments
...
- `trace_aditive::Bool`: if true performs a "all-to-all interaction" and else performs a "nearest-neighbor interaction".
"""
mutable struct RewardModulatedUpdate <: RewardModulatedUpdateRule
    reward::Float64
end
function (update_rule::RewardModulatedUpdate)(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency .* update_rule.reward
end

struct WeightDependentRewardModulatedUpdate <: RewardModulatedUpdateRule
    reward::Float64
end
function (update_rule::WeightDependentRewardModulatedUpdate)(network::SpikingNetwork, weight_update::AbstractArray)
    network.weight += weight_update .* network.adjacency * update_rule.reward .* network.weight
end

function set_reward(update_rule::RewardModulatedUpdateRule, new_reward::Float64)
    update_rule.reward = new_reward
end