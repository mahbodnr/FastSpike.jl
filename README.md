# FastSpike.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MahbodNr.github.io/FastSpike.jl/stable)

A Spiking Neural Network (SNN) simulation framework. FastSpike is designed to exploit the GPU memory in order to increase the speed of simulation as much as possible.

Please refer to `examples/` for more information. Python implementation: [FastSpike](https://github.com/mahbodnr/FastSpike)

# Getting started

## Installation

  ```bash
  pkg> add https://github.com/mahbodnr/FastSpike.jl
  ```

## Make a model

  ```julia
  using FastSpike
  model = Network(Izhikevich("fast spiking"))
  group_A = add_group!(model, 1000; name= "A")
  group_B = add_group!(model, 1000; name= "B")
  ```

### Heterogeneous Izhikevich neurons ([Izhikevich 2003](https://ieeexplore.ieee.org/document/1257420))

  ```julia
  r = rand(N, 1)
  a = reshape(0.02 .* excitatory + (0.02 .+ 0.08 .* r) .* inhibitory, (1, N))
  b = reshape(0.2 .* excitatory + (0.25 .- 0.05 .* r) .* inhibitory, (1, N)) 
  c = reshape((-65 .+ 15 .* r .^ 2) .* excitatory + (-65) .* inhibitory, (1, N)) 
  d = reshape((8 .- 6 .* r .^ 2) .* excitatory + 2 .* inhibitory, (1, N))

model = Network(
      neurons = Izhikevich(dt= dt, a= a, b= b, c= c, d= d, v_thresh= 30.0),
      learning_rule = STDP(A₊= 0.05, A₋= 0.05, τ₊= 20, τ₋= 20)
  )
  ```

## Run the model

  ```julia
  model = model |> gpu
  run!(
    model; 
    input_voltage= rand(Bool, batch_size, N) |> gpu
  )
  ```
