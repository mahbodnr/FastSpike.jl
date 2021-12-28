gpu(x::Nothing) = x
gpu(x::AbstractArray) = CuArray(x)

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
    )
end