"""
This file contains the classes to create the genome of an individual
"""
from random import randint, seed, uniform, choices, choice
from typing import Any
from conf import *
from activations import Activations
seed()


class LinkGene:
    def __init__(self, begin: int, end: int, innovation: int, weight: float = uniform(-1, 1)) -> None:
        """
        A gene that encodes a connection.

        :param begin: innovation number of the start neuron
        :param end: innovation number of the target neuron
        :param innovation: innovation number of the gene
        """
        self.end: int = end
        self.begin: int = begin
        self.weight: float = weight                               # weight of the connection
        self.enabled: bool = True                                 # if the active is operating or not
        self.innovation: int = innovation                         # the innovation number of the gene

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, type(self)) and
                self.begin == other.begin and self.end == other.end)

    def __hash__(self):
        return hash(self.innovation)


class NodeGene:
    def __init__(self, innovation: int, layer: int) -> None:
        """
        A gene that encodes a node. May evolve to have an activation function if that setting is enabled.

        :param innovation: innovation number of the neuron gene. Note that it is different from the innovation of a Link
        gene
        :param layer: layer of the neuron.
        """
        self.innovation: int = innovation
        self.layer: int = layer

        if layer > 1:
            self.activation: Any = DefaultActivationFunction
        else:
            self.activation: Any = "None"

    def __eq__(self, other: Any):
        return isinstance(other, type(self)) and other.innovation == self.innovation


NodeCount: int = 0
LinkCount: int = 0
LinkGenes: list[LinkGene] = []      # All the link genes


class Genome:
    NodeGenes: tuple = tuple()
    LinkGenes: tuple = tuple()
    Disabled: list = list()
    NodeCount: int = 0

    # noinspection PyPep8Naming
    def __init__(self, inputs: int, outputs: int) -> None:
        """
        The Genome of an individual. This class creates an object containing the information to build a network.

        :param inputs: number of input nodes
        :param outputs: number of output nodes
        """
        global NodeCount
        global LinkCount
        self.inputs: int = inputs        # number of input nodes
        self.outputs: int = outputs      # number of output nodes

        # create the bias neuron if enabled
        if BiasNeurons:
            self.NodeGenes = self.NodeGenes + (NodeGene(1, 1), )
            self.NodeCount += 1

        # create input and output neurons
        for i in range(0, self.inputs+self.outputs):
            if i < self.inputs:
                self.NodeGenes = self.NodeGenes + (NodeGene(self.NodeCount + 1, 1), )
            else:
                self.NodeGenes = self.NodeGenes + (NodeGene(self.NodeCount + 1, -1), )

            self.NodeCount += 1

        # Pre connect bias neuron if bias enabled
        if BiasNeurons:
            for i in range(0, self.outputs):
                NewGene = LinkGene(1, self.outputs+i+self.inputs, i + 1)      # Creating a new gene

                # Check if the gene exists to avoid duplicates with different innovation numbers
                if NewGene not in LinkGenes:
                    LinkCount += 1
                    LinkGenes.append(NewGene)
                else:
                    NewGene.innovation = LinkGenes[LinkGenes.index(NewGene)].innovation

                self.LinkGenes = self.LinkGenes + (NewGene, )                       # Appending the new gene

        # Pre connect input neurons to output neurons if enabled
        if ConnectInputs:
            bias_neurons = self.NodeCount-self.outputs-self.inputs  # number of bias neurons

            for i in range(0+bias_neurons, self.inputs+bias_neurons):
                # Create a new gene
                NewGene = LinkGene(i+1, randint(1, self.outputs) + self.inputs + bias_neurons, LinkCount + 1)

                # Check if the gene exists to avoid duplicates with different innovation numbers
                if NewGene not in LinkGenes:
                    LinkCount += 1
                    LinkGenes.append(NewGene)
                else:
                    NewGene.innovation = LinkGenes[LinkGenes.index(NewGene)].innovation

                self.LinkGenes = self.LinkGenes + (NewGene, )                       # Appending the new gene

            self.inputs += bias_neurons

        self.layers = 2
        NodeCount = self.NodeCount

    def mutate(self, amount: int = 1) -> None:
        """
        Mutates the genome based on the configuration of the algorithm. The configuration is present on the conf file.

        :param amount: Number of mutations performed
        """
        # Separate the mutations and the chances in 2 different tuples
        mutations: tuple = tuple(MutationChances.keys())
        chances: tuple = tuple(MutationChances.values())

        # Choose a random mutation based on the relative chances
        mutation: str = str(*choices(mutations, chances, k=1))
        print(mutation)

        match mutation:
            case "weight":
                self.m_weight()

            case "remove_link":
                self.m_remove_link()

            case "add_link":
                self.m_add_link()

            case "remove_node":
                self.m_remove_node()

            case "add_node":
                self.m_add_node()

            case "activation":
                self.m_activation()

        if amount > 1:
            self.mutate(amount - 1)

    def m_weight(self):
        gene: LinkGene = choice(self.LinkGenes)
        gene.weight += uniform(-WeightPerturbingAmount, WeightPerturbingAmount)

    def m_remove_link(self):
        gene: LinkGene = choice(self.LinkGenes)
        while gene in self.Disabled:
            gene = choice(self.LinkGenes)
        gene.enabled = False
        self.Disabled.append(gene)

    def m_add_link(self, depth: int = 800):
        if depth == 0:
            raise RecursionError("Unable to create a new link. Depth limit exceeded!")

        global LinkCount
        # Choosing 2 valid nodes
        node1: NodeGene = choice(self.NodeGenes)
        while node1.layer == -1:
            node1: NodeGene = choice(self.NodeGenes)

        node2: NodeGene = choice(self.NodeGenes)
        while -1 < node2.layer <= node1.layer:
            node2: NodeGene = choice(self.NodeGenes)

        # Create a valid link or enabling an existing one
        link = LinkGene(node1.innovation, node2.innovation, LinkCount + 1)
        if link in self.LinkGenes and link not in self.Disabled:
            self.m_add_link(depth-1)
        elif link in self.LinkGenes:
            self.Disabled.remove(link)
            self.LinkGenes[self.LinkGenes.index(link)].enabled = True
        else:
            LinkCount += 1

            # Checking if the link already exists in this generation and adding it to the existing link genes otherwise
            if link in LinkGenes:
                link.innovation = LinkGenes[LinkGenes.index(link)].innovation
            else:
                LinkGenes.append(link)

            self.LinkGenes = self.LinkGenes + (link, )

    def m_remove_node(self):
        # Choosing a node from a hidden layer
        node: NodeGene = choice(self.NodeGenes)
        while node.layer == -1 or node.layer == 1:
            node: NodeGene = choice(self.NodeGenes)

        self.NodeGenes = tuple(n for n in self.NodeGenes if n != node)

        for link in self.LinkGenes:
            if link.begin == node.innovation or link.end == node.innovation:
                self.LinkGenes = tuple(l for l in self.LinkGenes if l != link)

        raise NotImplementedError("Remove node mutation not implemented")

    def m_add_node(self):
        global NodeCount
        global LinkCount
        # Picking a random link that is enabled
        gene: LinkGene = choice(self.LinkGenes)
        while gene in self.Disabled:
            gene: LinkGene = choice(self.LinkGenes)

        # Adding the node
        node: NodeGene = NodeGene(NodeCount + 1, self.NodeGenes[gene.begin - 1].layer + 1)
        NodeCount += 1
        self.NodeGenes = self.NodeGenes + (node, )

        # Adjust the links
        gene.enabled = False
        self.Disabled.append(gene)
        self.LinkGenes = self.LinkGenes + (LinkGene(gene.begin, node.innovation, LinkCount + 1, 1), )
        LinkCount += 1
        self.LinkGenes = self.LinkGenes + (LinkGene(node.innovation, gene.end, LinkCount + 1, gene.weight), )
        LinkCount += 1

    def m_activation(self):
        gene_index = randint(0, len(self.NodeGenes) - 1)
        activation = choice(Activations.enabled_functions)
        while activation == self.NodeGenes[gene_index].activation:
            activation = choice(Activations.enabled_functions)

        self.NodeGenes[gene_index].activation = activation

    def __str__(self) -> str:
        string: str = "Link Genes:\nbegin\t|\tend\t  |\tenabled\t|   innovation\n"
        string += "".join(f"  {gene.begin : <7} {gene.end : ^8} {gene.enabled : ^9}|{gene.innovation : >8}\n" for gene in self.LinkGenes)

        string += "\n" + "-" * 40 + "\n"

        string += "\nNode Genes:\nlayer\t|\tactivation\t|\tinnovation\n"
        string += "".join(f"  {gene.layer : <12} {str(gene.activation) : <9}|{gene.innovation : >8}\n" for gene in self.NodeGenes)

        return string
