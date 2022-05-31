export NeuronType, LIF, Izhikevich, NeuronGroup

abstract type NeuronType end

struct LIF <: NeuronType
    dt::Real
    v_thresh::Real
    v_rest::Real
    v_reset::Real
    refractory_period::Int
    voltage_decay_factor::Real

    function LIF(dt::Real)
        new(
            dt,
            -52.0,
            -65.0,
            -65.0,
            5,
            exp(-dt / 100)
        )
    end
end

struct Izhikevich <: NeuronType
    dt::Real
    a::Union{Real,AbstractMatrix}
    b::Union{Real,AbstractMatrix}
    c::Union{Real,AbstractMatrix}
    d::Union{Real,AbstractMatrix}
    v_thresh::Union{Real,AbstractMatrix}
end


function Izhikevich(type::String; dt=1)
    if type == "RS" || type == "regular spiking"
        Izhikevich(
            dt,
            0.02,
            0.2,
            -65.0,
            8.0,
            30.0
        )
    elseif type == "IB" || type == "intrinsically bursting"
        Izhikevich(
            dt,
            0.02,
            0.2,
            -55.0,
            4.0,
            30.0
        )
    elseif type == "CH" || type == "chattering"
        Izhikevich(
            dt,
            0.02,
            0.2,
            -50.0,
            2.0,
            30.0
        )
    elseif type == "FS" || type == "fast spiking"
        Izhikevich(
            dt,
            0.1,
            0.2,
            -65.0,
            2.0,
            30.0
        )
    elseif type == "LTS" || type == "low-threshold spiking"
        Izhikevich(
            dt,
            0.02,
            0.25,
            -65.0,
            2.0,
            30.0
        )
    elseif type == "TC" || type == "thalamo-cortical"
        Izhikevich(
            dt,
            0.02,
            0.25,
            -65.0,
            0.05,
            30.0
        )
    elseif type == "RZ" || type == "resonator"
        Izhikevich(
            dt,
            0.1,
            0.26,
            -65.0,
            2.0,
            30.0
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