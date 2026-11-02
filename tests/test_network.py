import numpy as np
from pynn import NeuralNetwork
from pynn.visualise import Visualise
from pynn.population import Population

def test_network():
    baseNet = NeuralNetwork(10,3,[4, 9, 5,])
    pop = Population(baseNet, 10)
    pop.RandomisePopulation()

    print("Loop starting...")
    for i in range(1000):
        print(f"pop{i} begining mutation")
        c1 = np.random.randint(1,9)
        c2 = np.random.randint(2,9)
        best = [pop.population[c1], pop.population[c2]]
        pop.MutateNewBatch(best)

    for network in pop.population:
        print(network)

    Visualise(pop.population[0], "detailed")