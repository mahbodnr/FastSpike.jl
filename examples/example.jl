using FastSpike
using LinearAlgebra: I

time = 100
layer_size = 10
inner_weight = 0.5
weight_scale = 5

net = Network(LIF(1), 1, STDP(1.0, 1.0, 20, 20))
g1 = add_group!(net, layer_size)
g2 = add_group!(net, layer_size)
w1 = transpose(reshape([1:layer_size^2;], layer_size, :)) / (layer_size^2 / weight_scale)
connect!(net, g1, g2, w1)
w2 = inner_weight * (ones(layer_size, layer_size) - I)
connect!(net, g2, g2, w2)
net.weight
for t = 0:time
    s = zeros(Bool, (1, layer_size * 2))
    s[1, t%layer_size+1] = 1
    v = zeros((1, layer_size * 2))
    run!(net, s, v)
    println(t, net.spikes[1:layer_size], net.spikes[layer_size+1:end], net.voltage[end])
end

net.weight[1:10, 11:end]
net.weight[11:end, 11:end]
