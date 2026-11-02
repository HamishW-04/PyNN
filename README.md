# PyNN
PyNN is a library for creating and excecuting neural networks in python. It uses NumPy for the maths and currently uses MatPlot and NetworkX for basic visualisation. It is currently a work-in-progress, currently functioning but still needs optimisation, polish, etc.
## Overview
### Features
- Mutations
    - Weights/biases
    - Structural
    - Activation Layer Changing
- Population management
- Basic Visualisation
    - Simplified: Basic shows the number of neurons in a layer and the activation function used.
    - Detailed: Shows all the neurons in the network along side their biases. Weights are also shown with the thickness corresponding to their value
### Planned Features
- Improved visualisation
    - Better visualisation for very large networks.
    - Visualisation of 'thoughts'. With a given input, showing the influence of each Weight and the output of each node.
- Save and load using pathlib paths.
- Finish `Evaluate()` to take a fitness function so `fitnessScores` is calculated automatically.
## Code Examples
### Creating a netowork
Creating a random network. Commonly used as a start point for training. This example creates a network with 5 inputs, 3 outputs and has two hidden layers (both with 4 layers). A `list[int]` is used to specify the hidden layers.
```python
network = NeuralNetwork(5, 3, [4, 4])
network.Randomise()
```
Run the network using the `Forward()` function.
```python
inputs = [1, 5, 2, 4, 3]
netowrk.Forward(inputs) # Outputs as a list of length: outputSize
```
Use the `Mutate()` function to randomly mutate a network. Functions for specific mutations are also available.
```python
network.Mutate() #O ptionally, you can specify mutationStrength and mutationProbability. Both should be floats, 0 -> 1

# Also available:
network.MutateWeights() # Can also take strength and probability params
network.MutateBiases() # Can also take strength and probability params
network.Activation()
network.MutateStructure()
```
You can save and load networks using the following:
```python
network.Save("My Network", "Networks Folder") # path is currently a str type

loadedNetwork = NueralNetwork.Load("Networks Folder/My Network")
```
A population can be created using the `Population()` class. This takes a network as a base, and population size. The entire population can be randomised or mutated. On initialisation, the population will randomise each network.
```python
pop = Population(network, 100)

pop.fitnessScores = [20, 40, 50...] # list[int], length of which equals population size

bestIDs = pop.SelectBest(10) # Selects IDs of the 10 best networks from the population using the scores
pop.MutateNewBatch(bestIDs) # Keeps the best, fills the remaining population as mutations of the best.
```
## License
PyNN is licensed under the MIT License. See [LICENSE](LICENSE) for details