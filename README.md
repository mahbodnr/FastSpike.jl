# FastSpike.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MahbodNr.github.io/FastSpike.jl/stable)

A Spiking Neural Network (SNN) simulation framework
FastSpike is designed to exploit the GPU memory in order to increase the speed of simulation as much as possible.
Python implementation: [FastSpike](https://github.com/mahbodnr/FastSpike)
Please refer to `examples/` for more information

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
