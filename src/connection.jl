struct Connection
    weight::AbstractArray
    adjacency::AbstractArray
end

function randomConnection(n_source::Int, n_target::Int, p::AbstractFloat;
    min_weight=0, max_weight=1, amplitude=nothing, seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    adjacency = rand(n_source, n_target) .< p
    adjacency[adjacency+2I.>1] .= false
    if isnothing(amplitude)
        weights = rand(n_source, n_target) .* (max_weight - min_weight) .+ min_weight
    else
        weights = ones(n_source, n_target) .* amplitude
    end
    weights .*= adjacency
    return Connection(weights, adjacency)
end

function randomConnection(n::Int, p::AbstractFloat; min_weight=0, max_weight=1,
    amplitude=nothing, seed=nothing)
    randomConnection(n, n, p; min_weight, max_weight, amplitude, seed)
end



function EI(Nₑ::Integer, Nᵢ::Integer;
    E_E::Tuple=(0, 1), E_I::Tuple=(0, 1), I_E::Tuple=(0, 1), I_I::Tuple=(0, 1),
    adjacency::Union{Nothing,AbstractMatrix}=nothing, shuffle::Bool=false, seed=nothing
)
    excitatory = [ones(Bool, Nₑ); zeros(Bool, Nᵢ)]
    if shuffle
        if !isnothing(seed)
            Random.seed!(seed)
        end
        shuffle!(excitatory)
    end
    inhibitory = .!excitatory

    min_weight = (excitatory * transpose(excitatory)) .* E_E[1]
    min_weight += (excitatory * transpose(inhibitory)) .* E_I[1]
    min_weight -= (inhibitory * transpose(excitatory)) .* abs(I_E[2])
    min_weight -= (inhibitory * transpose(inhibitory)) .* abs(I_I[2])

    max_weight = (excitatory * transpose(excitatory)) .* E_E[2]
    max_weight += (excitatory * transpose(inhibitory)) .* E_I[2]
    max_weight -= (inhibitory * transpose(excitatory)) .* abs(I_E[1])
    max_weight -= (inhibitory * transpose(inhibitory)) .* abs(I_I[1])

    if !isnothing(adjacency)
        min_weight .*= adjacency
        max_weight .*= adjacency
    end

    return excitatory, inhibitory, min_weight, max_weight
end


function EI(n::Integer, EI_rate::AbstractFloat;
    E_E::Tuple=(0, 1), E_I::Tuple=(0, 1), I_E::Tuple=(0, 1), I_I::Tuple=(0, 1),
    adjacency::Union{Nothing,AbstractMatrix}=nothing, shuffle::Bool=false, seed=nothing
)
    if EI_rate > 1
        Nₑ = Int(ceil(EI_rate * n / (EI_rate + 1)))
    else
        Nₑ = Int(ceil(EI_rate * n))
    end
    Nᵢ = n - Nₑ
    return EI(Nₑ, Nᵢ; E_E, E_I, I_E, I_I, adjacency, shuffle, seed)
end
