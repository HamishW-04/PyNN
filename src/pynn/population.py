from pynn import NeuralNetwork
import numpy as np
import warnings

class Population:
    def __init__(self, startPoint:NeuralNetwork, popSize:int = 100):
        self.startPoint = startPoint
        self.popSize = popSize
        self.fitnessScores = []

        self.population = []
        for i in range(popSize):
            self.population.append(startPoint.Copy())

        self.RandomisePopulation()


    def RandomisePopulation(self):
        for network in self.population:
            network.Randomise()

    def MutateNewBatch(self, best:list):
        if len(best) > self.popSize:
            warnings.warn("Number of best networks must be smaller than the population size.")
            return
        
        newBatch = []

        for network in best:
            newBatch.append(network.Copy())

        while len(newBatch) < self.popSize:
            network = np.random.choice(best)
            newNet = network.Mutate()
            newBatch.append(newNet)

        self.population = newBatch

    def Evaluate(self, fitnessFunction):
        self.fitnessScores = []

        for network in self.population:
            score = fitnessFunction(1)
            self.fitnessScores.append(score)
        
    def SelectBest(self, num:int) -> list:
        bestIDs = np.argsort(self.fitnessScores)[-num:]
        return [self.population[i] for i in bestIDs]

    def DummyFitnessFunction(num:int) -> float:
        return 10.0 * num