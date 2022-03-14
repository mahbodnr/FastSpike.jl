cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = cu(x)

function gpu(net::Network)
    return Network(
        net.neurons,
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
        net.neurons,
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
        net.neurons,
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
        net.neurons,
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