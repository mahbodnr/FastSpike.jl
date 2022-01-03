export STDP

abstract type LearningRule end

"""
trace_aditive: if true performs a "all-to-all interaction" and else performs a "nearest-neighbor interaction". See: http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
"""
struct STDP <: LearningRule
    A₊::Real
    A₋::Real
    τ₊::Real
    τ₋::Real
    trace_scale::Real
    traces_additive::Bool
end

STDP() = STDP(1, 1, 10, 10, 1, false)

STDP(A₊::Real, A₋::Real, τ₊::Real, τ₋::Real; trace_scale=1, traces_additive= false) = STDP(
        A₊,
        A₋,
        τ₊,
        τ₋,
        trace_scale,
        traces_additive
    )