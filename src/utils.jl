using CUDA

function pad1D(X::CuArray, n::Int)
        return [X CUDA.zeros(eltype(X), (size(X,1),n))]
end

function pad2D(X::CuArray, n::Int)
        return [X CUDA.zeros(eltype(X), (size(X,1),n)); CUDA.zeros(eltype(X), (n,n+size(X,2)))]
end
