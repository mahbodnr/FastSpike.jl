using SafeTestsets

@safetestset "Networks Types" begin
    include("networks.jl")
end
@safetestset "Learning Rules" begin
    include("learning.jl")
end
#TODO:
# @safetestset "Neuron Types" begin include("neuron_type.jl") end