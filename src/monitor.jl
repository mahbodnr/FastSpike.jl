mutable struct Monitor
    network::Union{Network, DelayNetwork}
    spikes::AbstractArray
    voltage::AbstractArray
    recovery::AbstractArray
    # e₊::AbstractArray
    # e₋::AbstractArray
end
mutable struct WeightMonitor
    network::Union{Network, DelayNetwork}
    spikes::AbstractArray
    voltage::AbstractArray
    weight::AbstractArray
end

function Monitor(network::Union{Network,DelayNetwork}; record_weight = false)
    if record_weight
        return WeightMonitor(network, [], [], [])
    else
        return Monitor(network, [], [], [])
    end
end

function record!(monitor::Monitor)
    push!(monitor.spikes, monitor.network.spikes |> cpu)
    push!(monitor.voltage, monitor.network.voltage |> cpu)
    if typeof(monitor.network.neurons) <: Izhikevich
        push!(monitor.recovery, monitor.network.recovery |> cpu)
    end
    return
end

function record!(monitor::WeightMonitor)
    push!(monitor.spikes, monitor.network.spikes |> cpu)
    push!(monitor.voltage, monitor.network.voltage |> cpu)
    push!(monitor.weight, monitor.network.weight |> cpu)
    return
end

function Base.getindex(monitor::Monitor, idx::Union{UnitRange{Int},Vector{Int}})
    return Monitor(monitor.network[idx], monitor.spikes[:, :, idx], monitor.voltage[:, :, idx])
end

function Base.getindex(monitor::WeightMonitor, idx::Union{UnitRange{Int},Vector{Int}})
    return WeightMonitor(monitor.network[idx], monitor.spikes[:, :, idx], monitor.voltage[:, :, idx], monitor.WeightMonitor[:, idx, idx])
end

function save(monitor::Union{Monitor,WeightMonitor}, filename::AbstractString)
    save_object(filename, monitor)
end

function PSP(monitor::WeightMonitor; time = :, from = :, to = :)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    weight_array = convert(Array, VectorOfArray(monitor.weight)) #size: w1, w2, time
    spikes_array = convert(Array, VectorOfArray(monitor.spikes)) #size: batch_size, #neurons, time
    spikes_from_group = zeros(size(spikes_array))
    spikes_from_group[:, from, :] .= spikes_array[:, from, :]

    total_PSP = spikes_from_group ⊠ weight_array #size : batch_size, #neurons, tim
    return PermutedDimsArray(total_PSP[:, to, time], (3, 1, 2)) #size(PSP): time, batch_size, #neurons
end


function EPSP(monitor::WeightMonitor; time = :, from = :, to = :)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    pos(x) = ifelse(x > 0, x, 0)
    pos_weight = map(pos, convert(Array, VectorOfArray(monitor.weight))) #size: w1, w2, time
    spikes_array = convert(Array, VectorOfArray(monitor.spikes)) #size: batch_size, #neurons, time
    spikes_from_group = zeros(size(spikes_array))
    spikes_from_group[:, from, :] .= spikes_array[:, from, :]
    total_EPSP = spikes_from_group ⊠ pos_weight #size : batch_size, #neurons, time 
    return PermutedDimsArray(total_EPSP[:, to, time], (3, 1, 2)) #size(PSP): time, batch_size, #neurons
end

function IPSP(monitor::WeightMonitor; time = :, from = :, to = :)
    if typeof(from) == NeuronGroup
        from = from.idx
    end
    if typeof(to) == NeuronGroup
        to = to.idx
    end
    neg(x) = ifelse(x < 0, x, 0)
    neg_weight = map(neg, convert(Array, VectorOfArray(monitor.weight))) #size: w1, w2, time
    spikes_array = convert(Array, VectorOfArray(monitor.spikes)) #size: batch_size, #neurons, time
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