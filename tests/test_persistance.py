from pynn import NeuralNetwork
from pathlib import Path

root = Path(__file__).resolve().parent.parent
savePath = root / "tests" / "networks"

network = NeuralNetwork(3, 2, [3])
network.Randomise()

print(network)
network.Save(savePath, "ExampleSave")

new = NeuralNetwork.Load(savePath / "ExampleSave.json")
print(new)