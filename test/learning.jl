using FastSpike
using Test
# TODO: check etwork output with a calculated ground-truth.
N = 100
batch_size = 5

@testset "STDP" begin
    connection = randomConnection(N, 0.1)
    excitatory, inhibitory, min_weight, max_weight = EI(
        N, 0.8; adjacency=connection.adjacency, shuffle=true
    )
    # Define Network
    net = Network(
        neurons=LIF(),
        learning_rule=STDP(A₊=1, A₋=1, τ₊=20, τ₋=20,
            min_weight=min_weight, max_weight=max_weight,
            update_rule=WeightDependentRewardModulated(0)
        ),
        batch_size=batch_size,
    )
    neurons = add_group!(net, N; name="neurons")
    connect!(net, neurons, neurons, connection)
    set_reward(net, 1)
    run!(
        net;
        input_voltage=ones((batch_size, N)),
        input_spikes=ones(Bool, (batch_size, N))
    )
    @test true
end


@testset "vSTDP" begin
    connection = randomConnection(N, 0.1)
    excitatory, inhibitory, min_weight, max_weight = EI(
        N, 0.8; adjacency=connection.adjacency, shuffle=true
    )
    # Define Network
    net = Network(
        neurons=LIF(),
        learning_rule=vSTDP(A₊=1, A₋=1, τ₊=20, τ₋=20, τₓ=10,
            θ₋=10, θ₊=10, min_weight=min_weight, max_weight=max_weight,
        ),
        batch_size=batch_size,
    )
    neurons = add_group!(net, N; name="neurons")
    connect!(net, neurons, neurons, connection)
    run!(
        net;
        input_voltage=ones((batch_size, N)),
        input_spikes=ones(Bool, (batch_size, N))
    )
    @test true
end