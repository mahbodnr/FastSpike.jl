cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::AbstractArray) = cu(x)

function cpu(structure::Union{Network,DelayNetwork,NeuronType,LearningRule})
    return typeof(structure)(
        [cpu(getfield(structure, property)) for property in propertynames(structure)]...
    )
end

function gpu(structure::Union{Network,DelayNetwork,NeuronType,LearningRule})
    return typeof(structure)(
        [gpu(getfield(structure, property)) for property in propertynames(structure)]...
    )
end