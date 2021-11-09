abstract type LearningRule end

struct STDP{T <: Real} <: LearningRule
    A₊::T
    A₋::T
    τ₊::T
    τ₋::T
end
