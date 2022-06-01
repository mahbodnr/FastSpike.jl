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

function record!(monitor::Monitor, network::Network)
    push!(monitor.spikes, network.spikes |> copy |> cpu)
    push!(monitor.voltage, network.voltage |> copy |> cpu)
    if typeof(network.neurons) <: Izhikevich
        push!(monitor.recovery, network.recovery |> copy |> cpu)
    end
    return
end

function record!(monitor::WeightMonitor, network::Network)
    push!(monitor.spikes, network.spikes |> copy |> cpu)
    push!(monitor.voltage, network.voltage |> copy |> cpu)
    push!(monitor.weight, network.weight |> copy |> cpu)
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

function Base.get(monitor::MonitorActivity, field::Symbol)
    return convert(Array, VectorOfArray(getfield(monitor, field))) # size: batch_size, #neurons, time
end

function save(monitor::MonitorActivity, filename::AbstractString)
    save_object(filename, monitor)
end

function raster(monitor::MonitorActivity; batch=1)
    spikes_array = get(monitor, :spikes)
    return [(i[3], i[2]) for i in findall(spikes_array) if i[1] == batch]
end

function Base.getindex(monitor::MonitorActivity, idx::Union{UnitRange{Int},Vector{Int},Int})
    typeof(monitor)([getfield(monitor, i)[idx] for i in propertynames(monitor)]...)
end

function PSP(monitor::WeightMonitor; time=:, from=:, to=:)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    weight_array = get(monitor, :weight) #size: w1, w2, time
    spikes_array = get(monitor, :spikes) #size: batch_size, #neurons, time
    spikes_from_group = zeros(size(spikes_array))
    spikes_from_group[:, from, :] .= spikes_array[:, from, :]

    total_PSP = spikes_from_group ⊠ weight_array #size : batch_size, #neurons, tim
    return PermutedDimsArray(total_PSP[:, to, time], (3, 1, 2)) #size(PSP): time, batch_size, #neurons
end


function EPSP(monitor::WeightMonitor; time=:, from=:, to=:)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    pos(x) = ifelse(x > 0, x, 0)
    pos_weight = map(pos, get(monitor, :weight)) #size: w1, w2, time
    spikes_array = get(monitor, :spikes) #size: batch_size, #neurons, time
    spikes_from_group = zeros(size(spikes_array))
    spikes_from_group[:, from, :] .= spikes_array[:, from, :]
    total_EPSP = spikes_from_group ⊠ pos_weight #size : batch_size, #neurons, time 
    return PermutedDimsArray(total_EPSP[:, to, time], (3, 1, 2)) #size(PSP): time, batch_size, #neurons
end

function IPSP(monitor::WeightMonitor; time=:, from=:, to=:)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    neg(x) = ifelse(x < 0, x, 0)
    neg_weight = map(neg, get(monitor, :weight)) #size: w1, w2, time
    spikes_array = et(monitor, :spikes) #size: batch_size, #neurons, time
    spikes_from_group = zeros(size(spikes_array))
    spikes_from_group[:, from, :] .= spikes_array[:, from, :]
    total_EPSP = spikes_from_group ⊠ neg_weight #size : batch_size, #neurons, time 
    return PermutedDimsArray(total_EPSP[:, to, time], (3, 1, 2)) #size(PSP): time, batch_size, #neurons
end

# TODO: This part must be updated to support recent changes to the monitors
"""
function PSP(monitor::Monitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    total_spikes = dropdims(sum(monitor.spikes[time,:,:]; dims = 1); dims = 1) #size(s): batch_size, #neurons 
    spikes_from_group = zeros(size(total_spikes))
    spikes_from_group[:,from] .= total_spikes[:,from]
    return (spikes_from_group * monitor.network.weight)[:,to] #size(PSP): batch_size, #neurons
end

function EPSP(monitor::Monitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    pos(x) = ifelse(x>0 , x , 0)
    pos_weight = map(pos, monitor.network.weight)
    total_spikes = dropdims(sum(monitor.spikes[time,:,:]; dims = 1); dims = 1)
    spikes_from_group = zeros(size(total_spikes))
    spikes_from_group[:,from] .= total_spikes[:,from]
    return (spikes_from_group * pos_weight)[:,to]
end

function IPSP(monitor::Monitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    neg(x) = ifelse(x<0 , x , 0)
    neg_weight = map(neg, monitor.network.weight)
    total_spikes = dropdims(sum(monitor.spikes[time,:,:]; dims = 1); dims = 1)
    spikes_from_group = zeros(size(total_spikes))
    spikes_from_group[:,from] .= total_spikes[:,from]
    return (spikes_from_group * neg_weight)[:,to]
end
"""