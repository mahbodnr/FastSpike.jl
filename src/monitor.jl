using ..FastSpike: Network

mutable struct Monitor
    spikes::AbstractArray
    voltage::AbstractArray
    e₊::AbstractArray
    e₋::AbstractArray
end
Monitor() = Monitor([], [], [], [])

function record!(monitor::Monitor, network::Network)
    monitor.spikes = [monitor.spikes
        reshape(network.spikes, 1, size(network.spikes)...)]
    monitor.voltage = [monitor.voltage
        reshape(network.voltage, 1, size(network.voltage)...)]
    monitor.e₊ = [monitor.e₊
        reshape(network.e₊, 1, size(network.e₊)...)]
    monitor.e₋ = [monitor.e₋
        reshape(network.e₋, 1, size(network.e₋)...)]
end