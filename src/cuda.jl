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
        cpu(net.refractory),
        cpu(net.e₊),
        cpu(net.e₋),
        net.learning,
        net.groups
    )
end