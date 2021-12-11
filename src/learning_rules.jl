abstract type LearningRule end

struct STDP{T<:Real,R<:Real} <: LearningRule
    A₊::T
    A₋::T
    τ₊::R
    τ₋::R
end
