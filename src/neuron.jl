export NeuronType, LIF, Izhikevich, AdEx, NeuronGroup

abstract type NeuronType end

@with_kw struct LIF <: NeuronType
    dt::Real = 1
    v_thresh::Real = -52.0
    v_rest::Real = -65.0
    v_reset::Real = -65.0
    refractory_period::Int = 5
    voltage_time_constant::Real = 100
    voltage_decay_factor::Real = exp(-dt / voltage_time_constant)
    @assert voltage_decay_factor == exp(-dt / voltage_time_constant)
end

function add_group!(neurons::LIF, N::Int, batch_size::Int) end

@with_kw struct Izhikevich <: NeuronType
    dt::Real = 1
    a::Union{Real,AbstractMatrix} = 0.02
    b::Union{Real,AbstractMatrix} = 0.2
    c::Union{Real,AbstractMatrix} = -65.0
    d::Union{Real,AbstractMatrix} = 8.0
    v_thresh::Union{Real,AbstractMatrix} = 30.0
    runge_kutta_order::Int = 1
end

function Izhikevich(type::String; dt=1, runge_kutta_order=1)
    if type == "RS" || type == "regular spiking"
        Izhikevich(
            dt=dt,
            a=0.02,
            b=0.2,
            c=-65.0,
            d=8.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "IB" || type == "intrinsically bursting"
        Izhikevich(
            dt=dt,
            a=0.02,
            b=0.2,
            c=-55.0,
            d=4.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "CH" || type == "chattering"
        Izhikevich(
            dt=dt,
            a=0.02,
            b=0.2,
            c=-50.0,
            d=2.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "FS" || type == "fast spiking"
        Izhikevich(
            dt=dt,
            a=0.1,
            b=0.2,
            c=-65.0,
            d=2.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "LTS" || type == "low-threshold spiking"
        Izhikevich(
            dt=dt,
            a=0.02,
            b=0.25,
            c=-65.0,
            d=2.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "TC" || type == "thalamo-cortical"
        Izhikevich(
            dt=dt,
            a=0.02,
            b=0.25,
            c=-65.0,
            d=0.05,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    elseif type == "RZ" || type == "resonator"
        Izhikevich(
            dt=dt,
            a=0.1,
            b=0.26,
            c=-65.0,
            d=2.0,
            v_thresh=30.0,
            runge_kutta_order=runge_kutta_order,
        )
    else
        error("Unknown neuron type. Use one of 'RS' or 'regular spiking', 'IB' or 'intrinsically bursting', 'CH' or 'chattering', 'FS' or 'fast spiking', 'LTS' or 'low-threshold spiking', 'TC' or 'thalamo-cortical', 'RZ' or 'resonator'")
    end
end

function add_group!(neurons::Izhikevich, N::Int, batch_size::Int) end

@with_kw mutable struct AdEx <: NeuronType
    dt::Real = 1
    C::Union{Real,AbstractMatrix} = 281
    gₗ::Union{Real,AbstractMatrix} = 30
    Eₗ::Union{Real,AbstractMatrix} = -70.6
    Δₜ::Union{Real,AbstractMatrix} = 2
    Vₜ_rest::Union{Real,AbstractMatrix} = -50.4
    Vₜ_max::Union{Real,AbstractMatrix} = 30.4
    a::Union{Real,AbstractMatrix} = 4
    b::Union{Real,AbstractMatrix} = 80.5
    Iₛ::Union{Real,AbstractMatrix} = 400
    τz::Union{Real,AbstractMatrix} = 40
    τₐ::Union{Real,AbstractMatrix} = 144
    τᵥ::Union{Real,AbstractMatrix} = 50
    v_reset::Union{Real,AbstractMatrix} = -70.6
    # updating variables:
    v_thresh::Union{AbstractMatrix,Nothing} = nothing
    z::Union{AbstractMatrix,Nothing} = nothing
    wₐ::Union{AbstractMatrix,Nothing} = nothing
end

function add_group!(neurons::AdEx, N::Int, batch_size::Int)
    if isnothing(neurons.v_thresh)
        neurons.v_thresh = zeros(Float32, batch_size, 0)
        neurons.z = zeros(Float32, batch_size, 0)
        neurons.wₐ = zeros(Float32, batch_size, 0)
    end
    neurons.v_thresh = pad1D(neurons.v_thresh, N)
    neurons.z = pad1D(neurons.z, N)
    neurons.wₐ = pad1D(neurons.wₐ, N)
end

struct NeuronGroup
    n::Int
    idx::Union{UnitRange{Int64},Vector{Int64}}
end

Base.:+(A::NeuronGroup, B::NeuronGroup) = NeuronGroup(A.n + B.n, sort([A.idx; B.idx]))