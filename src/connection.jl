using LinearAlgebra: I
using Random

struct Connection
    weight::AbstractArray
    adjacency::AbstractArray
end

function randomConnection(n::Int, p::Float64; weight_min = 0, weight_max= 1, seed= nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    adjacency = rand(n, n) .> p
    adjacency -= adjacency .* (zeros(n ,n) + I) # remove recurrent connections
    weights = rand(n, n) .* adjacency .* (weight_max - weight_min) .+ weight_min
    return Connection(weights, adjacency)
end

function randomConnection(n_source::Int, n_target::Int, p::Float64; weight_min = 0, weight_max= 1, seed= nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    adjacency = rand(n_source, n_target) .> p
    weights = rand(n_source, n_target) .* adjacency .* (weight_max - weight_min) .+ weight_min
    return Connection(weights, adjacency)
end