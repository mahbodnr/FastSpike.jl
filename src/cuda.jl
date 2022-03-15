cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = cu(x)

function gpu(net::Network)
    return Network(
        gpu(net.neurons),
        net.batch_size,
        net.learning_rule,
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
        net.learning_rule,
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
        net.learning_rule,
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
        net.learning_rule,
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