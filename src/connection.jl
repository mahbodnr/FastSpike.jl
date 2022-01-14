struct Connection
    weight::AbstractArray
    adjacency::AbstractArray
end

function randomConnection(n::Int, p::AbstractFloat; min_weight = 0, max_weight= 1, seed= nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    adjacency = rand(n, n) .< p
    adjacency -= adjacency .* (zeros(n ,n) + I) # remove recurrent connections
    weights = rand(n, n) .* (max_weight - min_weight) .+ min_weight
    weights .*= adjacency
    return Connection(weights, adjacency)
end

function randomConnection(n_source::Int, n_target::Int, p::AbstractFloat; min_weight = 0, max_weight= 1, seed= nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    adjacency = rand(n_source, n_target) .< p
    weights = rand(n_source, n_target) .* (max_weight - min_weight) .+ min_weight
    weights .*= adjacency
    return Connection(weights, adjacency)
end

function EI(n::Integer, EI_rate::AbstractFloat;
    E_E::Tuple= (0,1), E_I::Tuple= (0,1), I_E::Tuple= (0,1), I_I::Tuple= (0,1),
    seed = nothing
    )
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    if EI_rate>1
        EI_rate = EI_rate/(EI_rate+1)
    end
    excitatory = rand(n) .< EI_rate
    inhibitory = .!excitatory

    min_weight =  (excitatory * transpose(excitatory)) .* E_E[1]
    min_weight += (excitatory * transpose(inhibitory)) .* E_I[1]
    min_weight -= (inhibitory * transpose(excitatory)) .* abs(I_E[2])
    min_weight -= (inhibitory * transpose(inhibitory)) .* abs(I_I[2])

    max_weight =  (excitatory * transpose(excitatory)) .* E_E[2]
    max_weight += (excitatory * transpose(inhibitory)) .* E_I[2]
    max_weight -= (inhibitory * transpose(excitatory)) .* abs(I_E[1])
    max_weight -= (inhibitory * transpose(inhibitory)) .* abs(I_I[1])
    
    return min_weight, max_weight
end
