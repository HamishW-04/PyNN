import matplotlib.pyplot as plt
import networkx as nx
from .network import NeuralNetwork

def Visualise(network:NeuralNetwork, mode:str = "compact"):

    graph = nx.DiGraph()

    neuronCounts = [network.inputSize]
    for layer in network.layers:
        neuronCounts.append(layer.outputSize)

    if mode == "detailed":
        # Neuron = (layerID, neuronID)
        for layerID, neuronCount in enumerate(neuronCounts):
            for neuronID in range(neuronCount):
                graph.add_node((layerID, neuronID))

        for layerID, layer in enumerate(network.layers):
            for inputNeuron in range(layer.inputSize):
                for outputNeuron in range(layer.outputSize):
                    graph.add_edge(
                        (layerID, inputNeuron),
                        (layerID + 1, outputNeuron),
                        weight=abs(layer.weights[outputNeuron, inputNeuron])
                    )
        positions = {}
        labels = {}
        for layerID, neuronCount in enumerate(neuronCounts):
            for neuronID in range(neuronCount):
                x = layerID
                y = (neuronCount -1) / 2  - neuronID
                positions[(layerID, neuronID)] = (x,y)
                if layerID == 0:
                    labels[(layerID, neuronID)] = "Input"
                else:
                    bias = network.layers[layerID-1].biases[neuronID, 0]
                    labels[(layerID, neuronID)] = f"{bias:.2f}"
                

        plt.figure(figsize=(10,6))
        edgeWidths=[graph[u][v]["weight"] for u, v in graph.edges]
        nx.draw(
            graph,
            pos=positions,
            node_size=1000,
            arrows=True,
            width=edgeWidths,
            labels=labels
        )

        plt.show()

    elif mode == "compact":
        positions = {}
        for layerID in range(len(neuronCounts)):
            graph.add_node(layerID)
            positions[layerID] = (layerID, 0)

        labels = {}
        for layerID, neuronCount in enumerate(neuronCounts):
            if layerID == 0:
                labels[layerID] = f"Input\n{neuronCount} neurons"
            else:
                activation = network.layers[layerID - 1].activationFunc
                labels[layerID] = f"{neuronCount} neurons\n{activation.name}"

        plt.figure(figsize=(10,6))
        nx.draw(
            graph,
            pos=positions,
            labels=labels,
            arrows=True
        )
        plt.show()


