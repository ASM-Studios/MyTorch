# How to create a neural network

## File configuration

To create a neural network, you have to specify it configuration in a configuration file.

The file must be a .json and follow this format:

### Layer

At the moment, you only have two type of layer:
- fully connected layer: `fully_connected`
- activation layer: `activation`

#### Fully connected

Parameters:
- the `input_size`: this is the number of entry taken by the layer (in a fully connected layer, this is generally the number of neuron of the previous layer)
- the `output_size`: this is the number of neuron of the layer

#### Activation

Parameters:
- the `activation` function

Here is the list of implemented activation functions:
- sigmoid
- tanh
- relu
- linear

### Loss

In this parameter, you have to specify the loss function which will be used by the neural network

Here is the list of implemented activation functions:
- mse
- cross_entropy: this function combine a softmax and cross entropy, take care to delete the last activation layer

## my_torch_generator
