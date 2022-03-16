include("../src/FastSpike.jl")
using Main.FastSpike
using LinearAlgebra: I
using ProgressMeter
using Plots

const time = 100 #seconds
const N = 1000
const group_size = 50


if isfile("network.jld2") # Load Network
    net = load("network.jld2")
    excitatory = (sum(net.weight; dims = 2).>0)[:, 1]
    inhibitory = (sum(net.weight; dims = 2).<0)[:, 1]
else # Or construct a new network
    connection = randomConnection(N, 0.1; amplitude = 5)
    excitatory, inhibitory, min_weight, max_weight = EI(
        N, 0.8;
        E_E = (0, 10), E_I = (0, 10), I_E = (-5, -5), I_I = (-5, -5),
        adjacency = connection.adjacency, shuffle = true, seed = 0
    )
    #Izhikevich neurons parameters
    a = reshape(0.02 .* excitatory + 0.1 .* inhibitory, (1, N))
    b = 0.2
    c = -65
    d = reshape(8 .* excitatory + 2 .* inhibitory, (1, N))
    # Define Network
    net = Network(
        Izhikevich(1, a, b, c, d, 30.0),
        1,
        STDP(0.1, 0.12, 20, 20; min_weight, max_weight)
    )
    # Add euron groups
    US = add_group!(net, group_size; name = "US")
    CS = add_group!(net, group_size; name = "CS")
    UR = add_group!(net, group_size; name = "UR")
    S = add_group!(net, N - group_size * 3; name = "S")
    neurons = US + CS + UR + S
    # connections in neural group
    connect!(net, neurons, neurons, connection)
end
# Define Monitor to record network activities
# monitor = Monitor(net)

net = net |> gpu
# Define training function
function train(time)
    @showprogress 1 "training " for t = 1:time
        # Make thalamic radom input
        random_input = zeros(1000, N)
        for t = 1:1000
            random_input[t, rand(1:N)] = 20
        end
        # Simulate for 1000 ms
        if t == 1 || t % 10 == 0 # Record each 10 seconds 
            spiked_neurons = []
            for ms = 1:1000 # each second
                run!(net; input_voltage = random_input[ms:ms, :] |> gpu)
                append!(spiked_neurons, [(ms, i[2]) for i in findall(net.spikes)])
            end
            # Save recordings
            plot(
                scatter(Tuple.(spiked_neurons), markersize = 1, c = :black, title = "time: $(t)s",),
                histogram(net.weight[excitatory, :][net.adjacency[excitatory, :]], title = "Excitatory weight histogram");
                layout = grid(2, 1, heights = [0.75, 0.25])
            )
            savefig("netork$(t)s.png")
            # save (overwrite) the model 
            save(net, "network.jl")
        else
            for ms = 1:1000 # each second
                run!(net; input_voltage = random_input[ms:ms, :] |> gpu)
            end
        end
    end
    save(net, "network.jl")
end

# Run
train(time)