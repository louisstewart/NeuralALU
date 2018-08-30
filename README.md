# Neural Aritmetic Logic Unit

Neural networks struggle to learn basic arithmetic functions (e.g. +, *). 
In the recent [DeepMind Paper](https://arxiv.org/pdf/1808.00508.pdf), they propose 
2 new neuron architectures, the Neural Accumulator and Neural Arithmetic Logic Unit
which are capable of allowing networks to learn these arithmetic functions.

The Neural Accumulator (NAC) allows the network to learn additive relations between
inputs, and the Neural Arithmetic Logic Unit (NALU) can learn the same
as well as multiplicative and exponential relations between inputs.

## Tensorflow Layers API

Tensorflow has a rather friendly API which a number of Neuron types have 
implemented. Cells already exist for most Dense, Conv and RNN cell types.

In this project, NAC and NALU cells have also been created, matching the Layers
API, for easy inclusion into other projects.

