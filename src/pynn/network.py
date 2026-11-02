import numpy as np
import json
from pathlib import Path
from .activations import ActivationFunction as AF
from .layer import Layer

import warnings

class NeuralNetwork:
    weightMutationMultiplier = 2
    biasMutationMultiplier = 2
    afMutationMultiplier = 0.5
    structureMutationMultiplier = 1

    def __init__(self, inputSize:int, outputSize:int, hiddenLayers:list[int], outputActivation:AF = AF.SOFTMAX):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = hiddenLayers.copy()
        self.outputActivation = outputActivation

        self.layers = []
        layerSizes = [inputSize, *hiddenLayers, outputSize]

        for i in range(len(layerSizes) -1):
            inSize = layerSizes[i]
            outSize = layerSizes[i+1]
            if i == len(layerSizes) - 2:
                activation = outputActivation
            else:
                activation = AF.LEAKY_RELU

            self.layers.append(Layer(inSize,outSize,activation))

    def __str__(self):
        string = "NeuralNetwork\n"
        string += f"  Input: {self.inputSize}\n"
        string += f"  Hidden: {self.hiddenLayers}\n"
        string += f"  Output: {self.outputSize}\n"
        string += f"  Output Activation: {self.outputActivation.name}\n"
        string += f"  Layers: {len(self.layers)}\n"

        for i, layer in enumerate(self.layers):
            string += (
                f"    Layer {i}: "
                f"{layer.inputSize} → {layer.outputSize} "
                f"| {layer.activationFunc.name}\n"
            )

        return string

    def Forward(self, x:np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1, 1)
        for layer in self.layers:
            x = layer.Forward(x)
        return x.flatten()

    def Randomise(self):
        for layer in self.layers:
            layer.Randomise()

    def Copy(self) -> NeuralNetwork:
        copy = NeuralNetwork(self.inputSize, self.outputSize, self.hiddenLayers, self.outputActivation)
        for i in range(len(self.layers)):
            copy.layers[i].weights = self.layers[i].weights.copy()
            copy.layers[i].biases = self.layers[i].biases.copy()
            copy.layers[i].activationFunc = self.layers[i].activationFunc

        return copy

    def Save(self, fileName:str, path:str = "Networks"):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filePath = path / fileName

        data = {
            "inputSize": self.inputSize,
            "outputSize": self.outputSize,
            "hiddenLayers": self.hiddenLayers,
            "outputActivation": self.outputActivation.name,
            "layers": []
        }

        for layer in self.layers:
            data["layers"].append({
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "activationFunc": layer.activationFunc.name 
            })

        with open(filePath, "w") as file:
            json.dump(data, file)

    @classmethod
    def Load(cls, fileName:str) -> NeuralNetwork:
        with open(fileName, "r") as file:
            data = json.load(file)
            network = cls(data["inputSize"], data["outputSize"], data["hiddenLayers"], AF[data["outputActivation"]])

            for i, layerData in enumerate(data["layers"]):
                network.layers[i].weights = np.array(layerData["weights"])
                network.layers[i].biases = np.array(layerData["biases"])
                network.layers[i].activationFunc = AF[layerData["activationFunc"]]

            return network

    def AddNeuron(self, layerID:int):
        if layerID >= 0 and layerID != (len(self.layers)-1):
            self.layers[layerID].AddOutput()
            self.layers[layerID+1].AddInput()
            self.hiddenLayers[layerID] += 1
        else:
             warnings.warn("Neuron was not added because the specified layer is the input or output layer.")

    def RemoveNeuron(self, layerID:int, neuronID:int):
        if layerID >= 0 and layerID != (len(self.layers)-1):
            self.layers[layerID].RemoveOutput(neuronID)
            self.layers[layerID+1].RemoveInput(neuronID)
            self.hiddenLayers[layerID] -= 1
        else:
            warnings.warn("Neuron was not removed because the specified layer is the input or output layer.")

    def AddLayer(self, layerID:int, mode:str = "empty", offsetStrength:float = 0.1):
        """
            Mode Options:
            - "empty"   : Identity weights and zero biases.
            - "random"  : Random weights and biases.
            - "offset"  : Slightly modify the neighbouring layer.
        """
        if layerID < len(self.layers) -2 and layerID >= 0:
            inputs = self.layers[layerID].outputSize
            outputs = self.layers[layerID].outputSize

            if mode == "empty":
                weights = np.identity(outputs)
                biases = np.zeros((outputs, 1))
            elif mode == "random":
                weights = np.random.randn(outputs, outputs)
                biases = np.random.randn(outputs,1)
            elif mode == "offset":
                weights = np.identity(outputs) + np.random.randn(outputs, outputs) * offsetStrength
                biases = np.random.randn(outputs, 1) * offsetStrength
            else:
                weights = np.identity(outputs)
                biases = np.zeros((outputs, 1))
                warnings.warn("New layer used empty mode as the mode entered was invalid")
            
            newLayer = Layer(inputs,outputs, AF.LEAKY_RELU)

            newLayer.weights = weights
            newLayer.biases = biases

            self.layers.insert(layerID + 1, newLayer)

            self.hiddenLayers.insert(layerID+1, outputs)
        else:
             warnings.warn("Layer Not added as not in valid range.")


    def Mutate(self, mutationStrength:float = 0.05, mutationProbability:float = 0.1) -> NeuralNetwork:
        child = self.Copy()
        mutated = False
        while not mutated:
            if np.random.random() < self.weightMutationMultiplier * mutationProbability:
                child.MutateWeights(mutationStrength, mutationProbability)
                mutated = True

            if np.random.random() < self.biasMutationMultiplier * mutationProbability:
                child.MutateBiases(mutationStrength, mutationProbability)
                mutated = True

            if np.random.random() < self.afMutationMultiplier * mutationProbability:
                child.MutateActivation()
                mutated = True

            if np.random.random() < self.structureMutationMultiplier * mutationProbability:
                child.MutateStructure()
                mutated = True
                    
        return child

    def MutateWeights(self, mutationStrength:float = 0.05, mutationProbability:float = 0.1):
        for layer in self.layers:
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    if np.random.random() <= mutationProbability:
                        mutation = max(layer.weights[i,j]*mutationStrength, 0.01)
                        layer.weights[i,j] += (np.random.choice([1, -1]) * mutation)

    def MutateBiases(self, mutationStrength:float = 0.05, mutationProbability:float = 0.1):
        for layer in self.layers:
            for i in range(len(layer.biases)):
                if np.random.random() <= mutationProbability:
                    mutation = max(layer.biases[i] * mutationStrength, mutationStrength/5)
                    layer.biases[i] +=  (np.random.choice([1, -1]) * mutation)

    def MutateActivation(self):
        layerID = np.random.randint(0, len(self.layers)-1)
        currentActivation = self.layers[layerID].activationFunc
        possibleActivations = [activation for activation in AF if activation != currentActivation]
        newActivation = np.random.choice(possibleActivations)
        self.layers[layerID].activationFunc = newActivation
        
        if layerID == len(self.layers) - 1:
            self.outputActivation = newActivation

    def MutateStructure(self):
        mutated = False
        while not mutated:
            choice = np.random.random()
            if choice <= 0.5:
                self.AddNeuron(np.random.randint(0,len(self.layers)-1))
                mutated = True
            elif choice <= 0.7:
                mode = np.random.choice(["empty", "random", "offset"])
                self.AddLayer(np.random.randint(0,len(self.layers)-2), mode)
                mutated = True
            else:
                layerID = np.random.randint(0,len(self.layers)-1)
                if self.layers[layerID].outputSize > 2:
                    neuronID = np.random.randint(0,self.layers[layerID].outputSize)
                    self.RemoveNeuron(layerID,neuronID)
                    mutated = True
             

