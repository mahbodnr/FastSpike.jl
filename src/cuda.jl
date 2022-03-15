cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = cu(x)

function gpu(net::Network)
    return Network(
        gpu(net.neurons),
        net.batch_size,
        gpu(net.learning_rule),
        gpu(net.weight),
        gpu(net.adjacency),
        gpu(net.spikes),
        gpu(net.voltage),
        gpu(net.recovery),
        gpu(net.refractory),
        gpu(net.e₊),
        gpu(net.e₋),
        net.learning,
        net.groups
    )
end

function cpu(net::Network)
    return Network(
        cpu(net.neurons),
        net.batch_size,
        cpu(net.learning_rule),
        cpu(net.weight),
        cpu(net.adjacency),
        cpu(net.spikes),
        cpu(net.voltage),
        cpu(net.recovery),
        cpu(net.refractory),
        cpu(net.e₊),
        cpu(net.e₋),
        net.learning,
        net.groups
    )
end

function gpu(net::DelayNetwork)
    return DelayNetwork(
        gpu(net.neurons),
        gpu(net.learning_rule),
        gpu(net.weight),
        gpu(net.adjacency),
        gpu(net.delay),
        gpu(net._delay),
        gpu(net.delayed_voltages),
        gpu(net.spikes),
        gpu(net.voltage),
        gpu(net.recovery),
        gpu(net.refractory),
        gpu(net.e₊),
        gpu(net.e₋),
        net.learning,
        net.groups
    )
end

function cpu(net::DelayNetwork)
    return DelayNetwork(
        cpu(net.neurons),
        cpu(net.learning_rule),
        cpu(net.weight),
        cpu(net.adjacency),
        cpu(net.delay),
        cpu(net._delay),
        cpu(net.delayed_voltages),
        cpu(net.spikes),
        cpu(net.voltage),
        cpu(net.recovery),
        cpu(net.refractory),
        cpu(net.e₊),
        cpu(net.e₋),
        net.learning,
        net.groups
    )
end

function gpu(neuron::Izhikevich)
    return Izhikevich(
        neuron.dt,
        gpu(neuron.a),
        gpu(neuron.b),
        gpu(neuron.c),
        gpu(neuron.d),
        gpu(neuron.v_thresh),
    )
end

function cpu(neuron::Izhikevich)
    return Izhikevich(
        neuron.dt,
        cpu(neuron.a),
        cpu(neuron.b),
        cpu(neuron.c),
        cpu(neuron.d),
        cpu(neuron.v_thresh),
    )
end

function gpu(learning_rule::STDP)
    return STDP(
        gpu(learning_rule.A₊),
        gpu(learning_rule.A₋),
        gpu(learning_rule.τ₊),
        gpu(learning_rule.τ₋),
        gpu(learning_rule.min_weight),
        gpu(learning_rule.max_weight),
        learning_rule.softbound,
        gpu(learning_rule.trace_scale),
        learning_rule.traces_additive,
    )
end

function cpu(learning_rule::STDP)
    return STDP(
        cpu(learning_rule.A₊),
        cpu(learning_rule.A₋),
        cpu(learning_rule.τ₊),
        cpu(learning_rule.τ₋),
        cpu(learning_rule.min_weight),
        cpu(learning_rule.max_weight),
        learning_rule.softbound,
        cpu(learning_rule.trace_scale),
        learning_rule.traces_additive,
    )
end
