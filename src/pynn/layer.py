import numpy as np
from . import activations
from .activations import ActivationFunction as AF

class Layer:
    def __init__(self, inputSize: int, outputSize:int, activationFunc:AF = AF.LEAKY_RELU):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activationFunc = activationFunc

        self.weights = np.zeros((outputSize,inputSize))
        self.biases = np.zeros((outputSize, 1))

    def Randomise(self):
        limit = np.sqrt(2.0 / self.inputSize)
        self.weights = np.random.randn(self.outputSize, self.inputSize) * limit
        self.biases = np.zeros((self.outputSize, 1))

    def Forward(self, x:np.ndarray) -> np.ndarray:
        y = self.weights @ x + self.biases
        function = activations.ActivationFunctions[self.activationFunc]
        return function(y)

    def AddOutput(self):
        newWeights = np.random.rand(1, self.inputSize)
        newBias = np.random.rand(1,1)

        self.weights = np.vstack((self.weights, newWeights))
        self.biases = np.vstack((self.biases, newBias))

        self.outputSize += 1

    def AddInput(self):
        assert self.weights.shape == (self.outputSize, self.inputSize)
        newWeights = np.random.rand(self.outputSize, 1)

        self.weights = np.hstack((self.weights, newWeights))
        self.inputSize += 1

    def RemoveOutput(self, neuronID:int):
        self.outputSize -= 1

        self.weights = np.delete(self.weights, neuronID, axis=0)
        self.biases = np.delete(self.biases, neuronID, axis=0)

    def RemoveInput(self, neuronID:int):
        self.inputSize -= 1

        self.weights = np.delete(self.weights, neuronID, axis=1)