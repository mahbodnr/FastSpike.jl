export STDP

abstract type LearningRule end

"""
trace_aditive: if true performs a "all-to-all interaction" and else performs a "nearest-neighbor interaction". See: http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
"""
struct STDP <: LearningRule #TODO: add "symetric::Bool" default field
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    min_weight::Union{Real,AbstractMatrix}
    max_weight::Union{Real,AbstractMatrix}
    softbound::Bool
    traces_additive::Bool
end

STDP() = STDP(1, 1, 10, 10, -Inf, Inf, false, 1, false)

STDP(A₊::Real, A₋::Real, τ₊::Real, τ₋::Real;
    min_weight = -Inf, max_weight = Inf, softbound = false,
    traces_additive = false) = STDP(
    A₊,
    A₋,
    τ₊,
    τ₋,
    min_weight,
    max_weight,
    softbound,
    traces_additive
)