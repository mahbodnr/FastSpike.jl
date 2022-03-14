using FastSpike
using LinearAlgebra: I
using Plots
using ProgressMeter

const time = 100
const N = 20
const input_dim = 5
# Define Network
net = Network(LIF(1), 2, STDP(1e-2, 1e-4, 20, 10; softbound = true, min_weight = 0, max_weight = 1))
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
# Define Monitor to record network activities
monitor = Monitor(net)
#plot weights histogram
plot_weights(activity, active_neurons) = display(
    plot(
        histogram(collect(Iterators.flatten(net.weight[input.idx, neurons.idx])), bins = 25),
        histogram(collect(Iterators.flatten(net.weight[neurons.idx, neurons.idx])), bins = 25),
        plot(activity),
        plot(scatter(active_neurons, markersize = 2, c = :black)),
        layout = (4, 1)
    )
)
# Define training function
function train(time)
    # Generate input spikes
    input_spikes = convert(Matrix{Bool}, [rand(0:1, time, input_dim) zeros(time, N)])
    active_neurons = []
    @showprogress 1 "training " for t = 1:time
        run!(net, input_spikes = reshape(input_spikes[t, :], 1, :))
        record!(monitor)
        append!(active_neurons, [(t, i[2]) for i in findall(net.spikes)])
        # plot_weights(sum(monitor.spikes[:, 1, neurons.idx], dims = 2), Tuple.(active_neurons))
    end
end
# Run
plot_weights([], [])
train(time)