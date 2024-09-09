"""
This file contains the NEAT algorithm implementation
"""
from random import choice
from network import Network
from gene import Genome
from conf import *

class Individual:
    def __init__(self, inputs: int, outputs: int) -> None:
        """
        An individual of the simulation. it contains both a genotype and a phenotype as well as a fitness score to
        determine if it will reproduce or not.

        :param inputs: number of input nodes
        :param outputs: number of output nodes
        """
        self.genome: Genome = Genome(inputs, outputs)
        self.network: Network = Network(self.genome.NodeGenes, self.genome.LinkGenes, inputs, outputs)
        self.fitness: None = None

    def evaluate(self, func) -> None:
        """
        Determines the fitness of the individual based on a given function. There is no default fitness function.

        :param func: function used to evaluate the individual
        """
        self.fitness: int = func(self)

    def forward(self, inputs: tuple[float | int]):
        if BiasNeurons:
            # Sets the value of the bias neuron to 1 if there is a bias neuron
            inputs = (1, ) + inputs

        return self.network.forward(inputs)


class NEAT:
    population = ()

    def __init__(self, inputs: int, outputs: int, population_amount: int = 1, generations: int | None = None) -> None:
        """
        The NEAT algorithm. This class creates an object that handles every thing needed in order to make the algorithm
        work. The only requirement is to provide a fitness function as there is no default.

        :param inputs: number of input nodes
        :param outputs: number of output nodes
        :param population_amount: the amount of individuals per generation
        :param generations: the max number of generations
        """
        self.inputs: int = inputs               # number of inputs
        self.outputs: int = outputs             # number of outputs
        self.generation: int = 1                 # current generation
        self.max_generation: int = generations   # max number of generations

        # initializing the population
        for i in range(0, population_amount):
            self.population = self.population + (Genome(inputs, outputs))

    def update(self, inputs: tuple[float | int]) -> None:
        """
        This function is responsible for training  handling everything the algorithm needs. This function should be could every program cycle.
        :param inputs:
        """

    def compute(self, inputs: tuple[float | int]) -> None:
        for individual in self.population:
            individual.forward(inputs)

    def crossover(self, parent1: Individual, parent2: Individual) -> None:
        # 25% chance of disabled genes be enabled
        raise NotImplementedError()