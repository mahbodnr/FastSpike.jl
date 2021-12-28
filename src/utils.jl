function load(filename::AbstractString)
        return load_object(filename)
    end

function pad1D(X::AbstractArray, n::Int)
        return [X zeros(eltype(X), (size(X, 1), n))]
end

function pad2D(X::AbstractArray, n::Int)
        return [X zeros(eltype(X), (size(X, 1), n)); zeros(eltype(X), (n, n + size(X, 2)))]
end

function regularActivity(N::Int, time::Int, frequency::Real)
        period = repeat([1; repeat([0],1000÷frequency-1)...], 1,N)
        return repeat(period, Int(ceil(time*frequency/1000)))[1:time, :]
end

function randomActivity(N::Int, time::Int, frequency::Real; seed= nothing)
        if !isnothing(seed)
                Random.seed!(seed)
        end
        return rand(time, N) .< (frequency/1000)
end