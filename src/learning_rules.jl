export STDP

abstract type LearningRule end

function regular_update(network, weight_update)
    network.weight += weight_update .* network.adjacency
    return
end

"""
`trace_aditive::Bool`: if true performs a "all-to-all interaction" and else performs a "nearest-neighbor interaction".
See: http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
"""
@kwdef mutable struct STDP <: LearningRule
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    min_weight::Union{Real,AbstractMatrix} = -Inf
    max_weight::Union{Real,AbstractMatrix} = Inf
    traces_additive::Bool = false
    update_rule::Function = regular_update
    e₊::Union{AbstractArray,Nothing} = nothing
    e₋::Union{AbstractArray,Nothing} = nothing
end

function add_group!(learning_rule::STDP, N::Int, batch_size::Int)
    if isnothing(learning_rule.e₊)
        learning_rule.e₊ = zeros(Int32, batch_size, 0)
        learning_rule.e₋ = zeros(Int32, batch_size, 0)
    end
    learning_rule.e₊ = pad1D(learning_rule.e₊, N)
    learning_rule.e₋ = pad1D(learning_rule.e₋, N)
end

function reset!(learning_rule::STDP)
    fill!(learning_rule.e₊, 0)
    fill!(learning_rule.e₋, 0)
end