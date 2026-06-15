"""
This file contains the classes to create the genome of an individual
"""
from random import randint, seed, uniform, choices, choice
from typing import Any
from conf import *
from activations import Activations
seed()


class LinkGene:
    def __init__(self, begin: int, end: int, innovation: int, weight: [float | None] = None) -> None:
        """
        A gene that encodes a connection.

        :param begin: innovation number of the start neuron
        :param end: innovation number of the target neuron
        :param innovation: innovation number of the gene
        :param weight: *optional. Weight of the link. If none is given a random number between -1 and 1 will be used
        """
        if weight is None:
            weight = uniform(-1, 1)

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
AllLinkGenes: list = list()      # All the link genes


class Genome:
    # noinspection PyPep8Naming
    def __init__(self, inputs: int | dict, outputs: int | dict, disabled: list = ()) -> None:
        """
        The Genome of an individual. This class creates an object containing the information to build a network.

        :param inputs: number of input nodes   | alternativily is a dict with the NodeGenes
        :param outputs: number of output nodes | alternativily is a dict with the LinkGenes
        """
        global NodeCount
        global LinkCount
        self.NodeGenes: dict = {}
        self.LinkGenes: dict = {}
        self.Disabled: list = list(disabled)
        self.NodeCount: int = 0

        if isinstance(inputs, int) and isinstance(outputs, int):
            self.inputs: int = inputs        # number of input nodes
            self.outputs: int = outputs      # number of output nodes

            # create the bias neuron if enabled
            if BiasNeurons:
                self.NodeGenes["1"] = NodeGene(1, 1)
                self.NodeCount += 1

            # create input and output neurons
            for i in range(0, self.inputs+self.outputs):
                if i < self.inputs:
                    self.NodeGenes[str(self.NodeCount + 1)] = NodeGene(self.NodeCount + 1, 1)
                else:
                    self.NodeGenes[str(self.NodeCount + 1)] = NodeGene(self.NodeCount + 1, -1)

                self.NodeCount += 1

            # Pre connect bias neuron if bias enabled
            if BiasNeurons:
                for i in range(0, self.outputs):
                    NewGene = LinkGene(1, self.NodeCount - i, i + 1)      # Creating a new gene

                    # Check if the gene exists to avoid duplicates with different innovation numbers
                    if NewGene not in AllLinkGenes:
                        LinkCount += 1
                        AllLinkGenes.append(NewGene)
                    else:
                        NewGene.innovation = AllLinkGenes[AllLinkGenes.index(NewGene)].innovation

                    self.LinkGenes[str(NewGene.innovation)] = NewGene                    # Appending the new gene

            # Pre connect input neurons to output neurons if enabled
            if ConnectInputs:
                bias_neurons = self.NodeCount-self.outputs-self.inputs  # number of bias neurons

                for i in range(0+bias_neurons, self.inputs+bias_neurons):
                    # Create a new gene
                    NewGene = LinkGene(i+1, randint(1, self.outputs) + self.inputs + bias_neurons, LinkCount + 1)

                    # Check if the gene exists to avoid duplicates with different innovation numbers
                    if NewGene not in AllLinkGenes:
                        LinkCount += 1
                        AllLinkGenes.append(NewGene)
                    else:
                        NewGene.innovation = AllLinkGenes[AllLinkGenes.index(NewGene)].innovation

                    self.LinkGenes[str(NewGene.innovation)] = NewGene                       # Appending the new gene

                self.inputs += bias_neurons

            self.layers = 2
            NodeCount = self.NodeCount

        else:
            self.NodeGenes = inputs
            self.LinkGenes = outputs
            self.NodeCount = len(self.NodeGenes)

    def mutate(self, amount: int = 1) -> None:
        """
        Mutates the genome based on the configuration of the algorithm. The configuration is present on the conf file.

        :param amount: Number of mutations performed
        """
        # Separate the mutations and the chances in 2 different tuples
        mutations: tuple = tuple(MutationChances.keys())
        chances: tuple = tuple(MutationChances.values())

        # Choose a random mutation based on the relative chances
        for i in range(0, amount):
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

    def m_weight(self):
        gene: LinkGene = self.LinkGenes[str(randint(1, LinkCount))]
        self.LinkGenes[str(gene.innovation)].weight += uniform(-WeightPerturbingAmount, WeightPerturbingAmount)

    def m_remove_link(self):
        enabled = [gene for n, gene in self.LinkGenes.items() if gene.enabled]
        if len(enabled) > 0:
            gene: LinkGene = choice(enabled)
            self.LinkGenes[str(gene.innovation)].enabled = False
            self.Disabled.append(gene.innovation)

    def m_add_link(self):
        global LinkCount
        # Choosing 2 valid nodes
        for i in range(0, 800):
            node1: NodeGene = choice(list(self.NodeGenes.values()))
            while node1.layer == -1:
                node1: NodeGene = choice(list(self.NodeGenes.values()))

            node2: NodeGene = choice(list(self.NodeGenes.values()))
            while -1 < node2.layer <= node1.layer:
                node2: NodeGene = choice(list(self.NodeGenes.values()))

            # Create a valid link or enabling an existing one
            link = LinkGene(node1.innovation, node2.innovation, LinkCount + 1)
            if link in self.LinkGenes.values() and link.innovation not in self.Disabled:
                continue
            elif link in self.LinkGenes.values():
                link.innovation = AllLinkGenes[AllLinkGenes.index(link)].innovation
                self.Disabled.remove(link.innovation)
                self.LinkGenes[str(link.innovation)].enabled = True
                return
            else:
                # Checking if the link already exists in this generation and adding it to the existing link genes otherwise
                if link in AllLinkGenes:
                    link.innovation = AllLinkGenes[AllLinkGenes.index(link)].innovation
                else:
                    AllLinkGenes.append(link)
                    LinkCount += 1

                self.LinkGenes[str(link.innovation)] = link
                return

        print("\033[33mWarning: Unable to add link\033[0m")

    def m_remove_node(self):
        # Choosing a node from a hidden layer
        node: NodeGene = choice(self.NodeGenes)
        while node.layer == -1 or node.layer == 1:
            node: NodeGene = choice(self.NodeGenes)

        self.NodeGenes = tuple(n for n in self.NodeGenes if n != node)

        for link in self.LinkGenes:
            if link.begin == node.innovation or link.end == node.innovation:
                self.LinkGenes = (l for l in self.LinkGenes if l != link)

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
        string: str = "\nNode Genes:\nlayer\t|\tactivation\t|\tinnovation\n"
        string += "".join(f"  {gene.layer : <12} {str(gene.activation) : <9}|{gene.innovation : >8}\n" for gene in self.NodeGenes.values())

        string += "\n" + "-" * 40 + "\n"

        string += "Link Genes:\nbegin\t|\tend\t  |\tenabled\t|   innovation\n"
        string += "".join(f"  {gene.begin : <7} {gene.end : ^8} {gene.enabled : ^9}|{gene.innovation : >8}\n" for gene in self.LinkGenes.values())

        return string
