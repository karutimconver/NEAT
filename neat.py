"""
This file contains the NEAT algorithm implementation
"""
from random import choice, random, randint
from copy import deepcopy, copy

from network import Network
from gene import Genome
from conf import *


class Individual:
    def __init__(self, inputs: int | Genome, outputs: int = None) -> None:
        """
        An individual of the simulation. it contains both a genotype and a phenotype as well as a fitness score to
        determine if it will reproduce or not.

        :param inputs: number of input nodes
        :param outputs: number of output nodes
        """
        if outputs:
            self.genome: Genome = Genome(inputs, outputs)
        else:
            self.genome: Genome = inputs
        self.network: Network = Network(self.genome.NodeGenes, self.genome.LinkGenes, inputs, outputs)
        self.fitness: int = 0

    def evaluate(self, func) -> None:
        """
        Determines the fitness of the individual based on a given function. There is no default fitness function.

        :param func: function used to evaluate the individual
        """
        self.fitness: int = func(self)

    def forward(self, inputs: tuple[float | int] | list[float | int]):
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        assert isinstance(inputs, tuple), "Unknown input format"

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
        self.inputs: int = inputs                # number of inputs
        self.outputs: int = outputs              # number of outputs
        self.generation: int = 1                 # current generation
        self.max_generation: int = generations   # max number of generations

        # initializing the population
        for i in range(0, population_amount):
            self.population = self.population + (Individual(inputs, outputs), )

    def update(self, inputs: tuple[float | int] | list[float | int]) -> None:
        """
        This function is responsible for training and handling everything the algorithm needs. This function should be called every program cycle.
        :param inputs: the inputs for the AI
        """
        raise NotImplementedError()

    def compute(self, inputs: tuple[float | int]) -> None:
        for individual in self.population:
            individual.forward(inputs)

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        fittest: Individual = parent1 if parent1.fitness > parent2.fitness else parent2
        if parent1.fitness == parent2.fitness:
            fittest = choice([parent1, parent2])

        other: Individual = parent1 if not parent1 is fittest else parent2

        offspringNodeGenes: dict = {}
        offspringLinkGenes: dict = {}
        disabled: list = []

        for gene in fittest.genome.NodeGenes.values():
            if gene.innovation in other.genome.NodeGenes.values():
                offspringNodeGenes[gene.innovation] = copy(choice([gene, other.genome.NodeGenes[gene.innovation]]))
            else:
                offspringNodeGenes[gene.innovation] = copy(gene)

        for gene in fittest.genome.LinkGenes.values():
            if gene.innovation in other.genome.LinkGenes.values():
                offspringLinkGenes[gene.innovation] = copy(choice([gene, other.genome.LinkGenes[gene.innovation]]))
            else:
                offspringLinkGenes[gene.innovation] = copy(gene)
            if not offspringLinkGenes[gene.innovation].enabled:
                disabled.append(gene.innovation)

        # 25% chance of disabled genes be enabled
        for id in disabled:
            if random() > ReenableGeneChance:
                offspringLinkGenes[id].enabled = True
                disabled.remove(id)

        genome1: Genome = Genome(offspringNodeGenes, offspringLinkGenes, copy(disabled))
        genome2: Genome = Genome(offspringNodeGenes, offspringLinkGenes, copy(disabled))

        genome1.mutate(randint(0, MaxMutationPerCrossover))
        genome2.mutate(randint(0, MaxMutationPerCrossover))

        child1 = Individual(genome1)
        child2 = Individual(genome2)

        return child1, child2
