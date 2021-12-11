using FastSpike
using LinearAlgebra: I
using Plots

const time = 100
const N = 20
const input_dim = 5
# Define Network
net = Network(LIF(1), 1, STDP(1.0, 1.0, 20, 20))
input = add_group!(net, input_dim)
neurons = add_group!(net, N)
# input to neurons
adjacency = rand(0:1, input_dim, N)
weights = rand(input_dim, N) .* adjacency
connect!(net, input, neurons, weights, adjacency)
# connections in neural group
adjacency = ones(N, N) - I
weights = rand(N, N) .* adjacency
connect!(net, neurons, neurons, weights, adjacency)
net.weight
# Generate input spikes
input_spikes = convert(Matrix{Bool}, [rand(0:1, time, input_dim) zeros(time, N)])
# Run
histogram(collect(Iterators.flatten(net.weight)))
for t = 1:time
    run!(net, input_spikes = reshape(input_spikes[t, :], 1, :))
    display(histogram(collect(Iterators.flatten(net.weight))))
end
