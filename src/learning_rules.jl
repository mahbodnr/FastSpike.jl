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
@kwdef struct STDP <: LearningRule
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    min_weight::Union{Real,AbstractMatrix} = -Inf
    max_weight::Union{Real,AbstractMatrix} = Inf
    traces_additive::Bool = false
    update_rule::Function = regular_update
end

