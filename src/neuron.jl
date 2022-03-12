export NeuronType, LIF, Izhikevich, NeuronGroup

abstract type NeuronType end

struct LIF <: NeuronType
    dt::Int
    v_thresh::Real
    v_rest::Real
    v_reset::Real
    refractory_period::Int
    voltage_decay_factor::Real

    function LIF(dt::Int)
        new(
            UInt(dt),
            -52.0,
            -65.0,
            -65.0,
            5,
            exp(-dt / 100)
        )
    end
end

struct Izhikevich <: NeuronType
    dt::Int
    a::Real
    b::Real
    c::Real
    d::Real
    v_thresh::Real

    function Izhikevich(type::String; dt = 1)
        if type == "RS" || type == "regular spiking"
            new(
                UInt(dt),
                0.02,
                0.2,
                -65.0,
                8.0,
                30.0
            )
        elseif type == "IB" || type == "intrinsically bursting"
            new(
                UInt(dt),
                0.02,
                0.2,
                -55.0,
                4.0,
                30.0
            )
        elseif type == "CH" || type == "chattering"
            new(
                UInt(dt),
                0.02,
                0.2,
                -50.0,
                2.0,
                30.0
            )
        elseif type == "FS" || type == "fast spiking"
            new(
                UInt(dt),
                0.1,
                0.2,
                -65.0,
                2.0,
                30.0
            )
        elseif type == "LTS" || type == "low-threshold spiking"
            new(
                UInt(dt),
                0.02,
                0.25,
                -65.0,
                2.0,
                30.0
            )
        elseif type == "TC" || type == "thalamo-cortical"
            new(
                UInt(dt),
                0.02,
                0.25,
                -65.0,
                0.05,
                30.0
            )
        elseif type == "RZ" || type == "resonator"
            new(
                UInt(dt),
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
end

struct NeuronGroup
    n::Int
    idx::Union{UnitRange{Int64},Vector{Int64}}
end

Base.:+(A::NeuronGroup, B::NeuronGroup) = NeuronGroup(A.n + B.n, sort([A.idx; B.idx]))