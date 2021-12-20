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

function STDP()
    STDP(
        1,
        1,
        10,
        10,
        1,
        false
    )
end

function STDP(A₊::Real, A₋::Real, τ₊::Real, τ₋::Real)
    STDP(
        A₊,
        A₋,
        τ₊,
        τ₋,
        1,
        false
    )
end