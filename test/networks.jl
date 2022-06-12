using FastSpike
using Test
# TODO: check etwork output with a calculated ground-truth.
N = 100
batch_size = 1

for network in [Network, DelayNetwork]
    @testset "$network of Homogeneous Izhikevich neurons" begin
        connection = randomConnection(N, 0.1)
        #Izhikevich neurons parameters
        a = 0.02
        b = 0.2
        c = -65
        d = 2
        # Define Network
        net = network(
            neurons=Izhikevich(dt=1, a=a, b=b, c=c, d=d, v_thresh=30.0),
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

    @testset "$network of Heterogeneous Izhikevich neurons" begin
        connection = randomConnection(N, 0.1)
        size_n1 = Int(N / 2)
        size_n2 = N - size_n1
        n1 = Array{Bool}([ones(size_n1); zeros(size_n2)])
        n2 = Array{Bool}([zeros(size_n1); ones(size_n2)])
        #Izhikevich neurons parameters
        r = rand(N, 1)
        a = reshape(0.02 .* n1 + (0.02 .+ 0.08 .* r) .* n2, (1, N))
        b = reshape(0.2 .* n1 + (0.25 .- 0.05 .* r) .* n2, (1, N))
        c = reshape((-65 .+ 15 .* r .^ 2) .* n1 + (-65) .* n2, (1, N))
        d = reshape((8 .- 6 .* r .^ 2) .* n1 + 2 .* n2, (1, N))
        # Define Network
        net = network(
            neurons=Izhikevich(dt=1, a=a, b=b, c=c, d=d, v_thresh=30.0),
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

    @testset "$network of LIF neurons" begin
        connection = randomConnection(N, 0.1)
        # Define Network
        net = network(
            neurons=LIF(),
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
end
