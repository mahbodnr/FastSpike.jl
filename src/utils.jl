function pad1D(X::AbstractArray, n::Int)
        return [X zeros(eltype(X), (size(X, 1), n))]
end

function pad2D(X::AbstractArray, n::Int)
        return [X zeros(eltype(X), (size(X, 1), n)); zeros(eltype(X), (n, n + size(X, 2)))]
end