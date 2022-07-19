export LearningRule, vSTDP, STDP

abstract type LearningRule end

"""
# STDP learning rule
Spike Timing Dependent Plasticity learning rule. See: http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
# Arguments
...
- `trace_aditive::Bool`: if true performs a "all-to-all interaction" and else performs a "nearest-neighbor interaction".
"""
@with_kw mutable struct STDP <: LearningRule
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    min_weight::Union{Real,AbstractMatrix} = -Inf
    max_weight::Union{Real,AbstractMatrix} = Inf
    traces_additive::Bool = false
    update_rule::UpdateRule = RegularUpdate()
    e₊::Union{AbstractArray,Nothing} = nothing
    e₋::Union{AbstractArray,Nothing} = nothing
end

function add_group!(learning_rule::STDP, N::Int, batch_size::Int)
    if isnothing(learning_rule.e₊)
        learning_rule.e₊ = zeros(Float32, batch_size, 0)
        learning_rule.e₋ = zeros(Float32, batch_size, 0)
    end
    learning_rule.e₊ = pad1D(learning_rule.e₊, N)
    learning_rule.e₋ = pad1D(learning_rule.e₋, N)
end

function reset!(learning_rule::STDP)
    fill!(learning_rule.e₊, 0)
    fill!(learning_rule.e₋, 0)
end

"""
# Voltage-based STDP
 Voltage-based STDP learning rule. See: https://www.nature.com/articles/nn.2479
# Arguments
...
"""
@with_kw mutable struct vSTDP <: LearningRule
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    τₓ::Real
    θ₋::Real
    θ₊::Real
    min_weight::Union{Real,AbstractMatrix} = -Inf
    max_weight::Union{Real,AbstractMatrix} = Inf
    ū₋::Union{AbstractArray,Nothing} = nothing
    ū₊::Union{AbstractArray,Nothing} = nothing
    x̄::Union{AbstractArray,Nothing} = nothing
end

function add_group!(learning_rule::vSTDP, N::Int, batch_size::Int)
    if isnothing(learning_rule.x̄)
        learning_rule.x̄ = zeros(Float32, batch_size, 0)
        learning_rule.ū₋ = zeros(Float32, batch_size, 0)
        learning_rule.ū₊ = zeros(Float32, batch_size, 0)
    end
    learning_rule.x̄ = pad1D(learning_rule.x̄, N)
    learning_rule.ū₋ = pad1D(learning_rule.ū₋, N)
    learning_rule.ū₊ = pad1D(learning_rule.ū₊, N)
end

function reset!(learning_rule::vSTDP)
    fill!(learning_rule.x̄, 0)
    fill!(learning_rule.ū₋, 0)
    fill!(learning_rule.ū₊, 0)
end