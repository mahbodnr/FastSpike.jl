mutable struct Monitor
    network::Network
    spikes::AbstractArray
    voltage::AbstractArray
    # e₊::AbstractArray
    # e₋::AbstractArray
end
mutable struct WeightMonitor
    network::Network
    spikes::AbstractArray
    voltage::AbstractArray
    weight::AbstractArray
end

function Monitor(network::Network; record_weight = false)
    if record_weight
        return WeightMonitor(network, [],[],[])
    else
        return Monitor(network, [],[])
    end
end

function record!(monitor::Monitor)
    push!(monitor.spikes, monitor.network.spikes)
    push!(monitor.voltage, monitor.network.voltage)
end

function record!(monitor::WeightMonitor)
    push!(monitor.spikes, monitor.network.spikes)
    push!(monitor.voltage, monitor.network.voltage)
    push!(monitor.weight, monitor.network.weight)
end

function Base.getindex(monitor::Monitor, idx::Union{UnitRange{Int},Vector{Int}})
    return Monitor(monitor.network[idx], monitor.spikes[:,:,idx], monitor.voltage[:,:,idx])
end

function Base.getindex(monitor::WeightMonitor, idx::Union{UnitRange{Int},Vector{Int}})
    return WeightMonitor(monitor.network[idx], monitor.spikes[:,:,idx], monitor.voltage[:,:,idx], monitor.WeightMonitor[:,idx,idx])
end

function save(monitor::Union{Monitor, WeightMonitor}, filename::AbstractString)
    save_object(filename, monitor)
end

# TODO: This part must be updated to support recent changes to the monitors
"""
function PSP(monitor::WeightMonitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    spikes_from_group = zeros(size(monitor.spikes))
    spikes_from_group[:,:,from] .= monitor.spikes[:,:,from]

    total_PSP = (
        PermutedDimsArray(spikes_from_group, (2,3,1)) ⊠ PermutedDimsArray(monitor.weight, (2,3,1))
    ) #size : batch_size, #neurons, time 
    return PermutedDimsArray(total_PSP[:,to,time], (3,1,2)) #size(PSP): time, batch_size, #neurons
end


function EPSP(monitor::WeightMonitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    pos(x) = ifelse(x>0 , x , 0)
    pos_weight = map(pos, monitor.weight)

    spikes_from_group = zeros(size(monitor.spikes))
    spikes_from_group[:,:,from] .= monitor.spikes[:,:,from]

    total_EPSP = (
        PermutedDimsArray(spikes_from_group, (2,3,1)) ⊠ PermutedDimsArray(pos_weight, (2,3,1))
    ) #size : batch_size, #neurons, time 

    return PermutedDimsArray(total_EPSP[:,to,time], (3,1,2)) #size(PSP): time, batch_size, #neurons
end

function IPSP(monitor::WeightMonitor; time =:, from =:, to =:)
    if typeof(from) == NeuronGroup
            from = from.idx
    end
    if typeof(to) == NeuronGroup
            to = to.idx
    end
    neg(x) = ifelse(x<0 , x , 0)
    neg_weight = map(neg, monitor.weight)

    spikes_from_group = zeros(size(monitor.spikes))
    spikes_from_group[:,:,from] .= monitor.spikes[:,:,from]

    total_IPSP = (
        PermutedDimsArray(spikes_from_group, (2,3,1)) ⊠ PermutedDimsArray(neg_weight, (2,3,1))
    ) #size : batch_size, #neurons, time 

    return PermutedDimsArray(total_IPSP[:,to,time], (3,1,2)) #size(PSP): time, batch_size, #neurons
end


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