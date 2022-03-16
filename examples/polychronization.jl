"""
An implementation of "Polychronization: computation with spikes"
Izhikevich EM. Neural Comput. 2006 Feb;18(2):245-82. doi: 10.1162/089976606775093882. PMID: 16378515.
https://pubmed.ncbi.nlm.nih.gov/16378515/
"""
using FastSpike
using LinearAlgebra: I
using Plots
using ProgressMeter

const time = 10_000
const Nₑ = 800
const Nᵢ = 200
const N = Nₑ + Nᵢ
const p = 0.1
#Izhikevich neurons parameters
a = [0.02 * ones(1, Nₑ) 0.1 * ones(1, Nᵢ)]
b = 0.2
c = -65
d = [8 * ones(1, Nₑ) 2 * ones(1, Nᵢ)]
# form adjacency matrix
adjacency = rand(N, N) .< p
adjacency -= adjacency .* (zeros(N, N) + I) # remove recurrent connections
# Inhibitory/Excitetory weights: excitatory: 0 → 10, inhibitory: -5 (rigid)
min_weight, max_weight = EI(
    Nₑ, Nᵢ;
    E_E = (0, 10), E_I = (0, 10), I_E = (-5, -5), I_I = (-5, -5),
    adjacency = adjacency
)
# Define Network
net = DelayNetwork(
    Izhikevich(1, a, b, c, d, 30.0),
    STDP(0.1, 0.12, 20, 20; min_weight = min_weight, max_weight = max_weight)
)
neurons = add_group!(net, N)
# connections in neural group
weights = [6 * ones(Nₑ, Nₑ + Nᵢ); -5 * ones(Nᵢ, Nₑ + Nᵢ)]
weights .*= adjacency
delay = rand(1:20, N, N)
delay .*= adjacency
connect!(net, neurons, neurons, weights, adjacency, delay)
# Define Monitor to record network activities
monitor = Monitor(net)
# Define training function
function train(time)
    spiked_neurons = []
    @showprogress 1 "training " for t = 1:time
        random_input = zeros(1, N)
        random_input[rand(1:N)] = 20
        run!(net; input_voltage = random_input)
        # record!(monitor)
        append!(spiked_neurons, [(t, i[2]) for i in findall(net.spikes)])
        if t % 1000 == 0
            plot(
                scatter(Tuple.(spiked_neurons), markersize = 1, c = :black, title = "time: $(t÷1000)s",),
                histogram(net.weight[1:800, :][convert(Matrix{Bool}, net.adjacency)[1:800, :]], title = "Excitatory weight histogram");
                layout = grid(4, 1, heights=[.75, .25]),
            )
            spiked_neurons = []
        end
    end
    return
end
# Run
train(time)
