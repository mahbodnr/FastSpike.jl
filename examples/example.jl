using FastSpike
using CUDA

net = Network(LIF(1), 1)
g1 = add_group!(net, 3)
g2 = add_group!(net, 2)

g3 = add_group!(net, 2)

w = CUDA.ones((3,2)) .* 0.5
connect!(net, g1, g2, w)

w = CUDA.ones((3,2)) .* -0.5
adj = zeros((2,2))
adj[1,2]=1
connect!(net, g2, g3, w, CuArray(adj))

s = CUDA.ones(Bool, (1,7))
v = CUDA.ones((1,7))

# run!(net, s, v)
