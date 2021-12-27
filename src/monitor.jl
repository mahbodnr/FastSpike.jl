using ..FastSpike: Network
using JLD2

mutable struct Monitor
    network::Network
    spikes::AbstractArray
    voltage::AbstractArray
    # e₊::AbstractArray
    # e₋::AbstractArray
end
Monitor(network::Network) = Monitor(network, [], [])

function record!(monitor::Monitor)
    dims = size(monitor.network.spikes)
    monitor.spikes = [monitor.spikes
        reshape(monitor.network.spikes, 1, dims...)]
    monitor.voltage = [monitor.voltage
        reshape(monitor.network.voltage, 1, dims...)]
    # if !isnothing(monitor.network.e₊)
    #     monitor.e₊ = [monitor.e₊
    #         reshape(monitor.network.e₊, 1, dims...)]
    # end
    # if !isnothing(monitor.network.e₋)
    #     monitor.e₋ = [monitor.e₋
    #         reshape(monitor.network.e₋, 1, dims...)]
    # end
end

function Base.getindex(monitor::Monitor, idx::Union{UnitRange{Int64},Vector{Int64}})
    return Monitor(monitor.network, monitor.spikes[:,:,idx], monitor.voltage[:,:,idx])
end

function save(monitor::Monitor, filename::AbstractString)
    save_object(filename, monitor)
end