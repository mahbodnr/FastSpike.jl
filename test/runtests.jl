using SafeTestsets

@safetestset "Networks Types" begin
    include("networks.jl")
end
#TODO:
# @safetestset "Neuron Types" begin include("neuron_type.jl") end
# @safetestset "Neuron Types" begin include("learning.jl") end