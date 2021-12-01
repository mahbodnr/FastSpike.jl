using CUDA

gpu(x::Nothing) = x
gpu(x::AbstractArray) = CuArrays.cu(x)

function gpu(net::Network)
    neurons = net.neurons
    batch_size = net.batch_size
    learning_rule = net.learning_rule
    weight = gpu(net.weight)
    adjacency = gpu(net.adjacency)
    spikes = gpu(net.spikes)
    voltage = gpu(net.voltage)
    refractory = gpu(net.refractory)
    e₊ = gpu(net.e₊)
    e₋ = gpu(net.e₋)
    learning = net.learning

    return Network(
        neurons,
        batch_size,
        learning_rule,
        weight,
        adjacency,
        spikes,
        voltage,
        refractory,
        e₊,
        e₋,
        learning,
    )
end