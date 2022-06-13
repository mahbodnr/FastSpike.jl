export NeuronType, LIF, Izhikevich, NeuronGroup

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

struct NeuronGroup
    n::Int
    idx::Union{UnitRange{Int64},Vector{Int64}}
end

Base.:+(A::NeuronGroup, B::NeuronGroup) = NeuronGroup(A.n + B.n, sort([A.idx; B.idx]))