export NeuronType, LIF, NeuronGroup

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


struct NeuronGroup
    n::Int
    idx::UnitRange{Int64}
end
