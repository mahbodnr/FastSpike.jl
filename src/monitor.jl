abstract type MonitorActivity end

mutable struct Monitor <: MonitorActivity
    spikes::AbstractArray
    voltage::AbstractArray
    recovery::AbstractArray
    # e₊::AbstractArray
    # e₋::AbstractArray
end
mutable struct WeightMonitor <: MonitorActivity
    spikes::AbstractArray
    voltage::AbstractArray
    weight::AbstractArray
end

function Monitor(; record_weight=false)
    if record_weight
        return WeightMonitor([], [], [])
    else
        return Monitor([], [], [])
    end
end

function record!(monitor::Monitor, network::SpikingNetwork)
    push!(monitor.spikes, network.spikes |> copy)
    push!(monitor.voltage, network.voltage |> copy)
    if typeof(network.neurons) <: Izhikevich
        push!(monitor.recovery, network.recovery |> copy)
    end
    return
end

function record!(monitor::WeightMonitor, network::SpikingNetwork)
    push!(monitor.spikes, network.spikes |> copy)
    push!(monitor.voltage, network.voltage |> copy)
    push!(monitor.weight, network.weight |> copy)
    return
end

function reset!(monitor::Monitor)
    monitor.spikes = []
    monitor.voltage = []
    monitor.recovery = []
    return
end

function reset!(monitor::WeightMonitor)
    monitor.spikes = []
    monitor.voltage = []
    monitor.weight = []
    return
end

"""
Returns a specific field from monitor. The output shape is: (batch_size, #neurons, time)
# Examples
```julia-repl
julia> get(network, :spikes)
```
"""
function Base.get(monitor::MonitorActivity, field::Symbol)
    return convert(Array, VectorOfArray(getfield(monitor, field)))
end

function save(monitor::MonitorActivity, filename::AbstractString)
    save_object(filename, monitor)
end

function raster(monitor::MonitorActivity; batch=1)
    spikes_array = get(monitor, :spikes)
    return [(i[3], i[2]) for i in findall(spikes_array) if i[1] == batch]
end

function Base.getindex(monitor::MonitorActivity, idx::Union{UnitRange{Int},Vector{Int},Int})
    typeof(monitor)([getfield(monitor, property)[idx] for property in propertynames(monitor)]...)
end

