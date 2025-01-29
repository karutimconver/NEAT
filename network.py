"""
This file contains the classes to create the network itself.
"""
from typing import Any
from activations import ActivationFunctions
from copy import deepcopy

activations: ActivationFunctions = ActivationFunctions()


class Link:
    def __init__(self, begin: int, end: int, weight: float) -> None:
        """
        A connection between two different nodes of the network

        :param begin: id of the start node
        :param end: id of the target node
        :param weight: weight of the connection
        """
        self.weight: float = weight
        self.begin: int = begin
        self.end: int = end


class Node:
    input: list = []
    output: int | float = 0

    def __init__(self, activation: str, _id: int) -> None:
        """
        A node of the neural network. Takes some inputs and processes the output.
        Both the inputs and the output are carried trough links.

        :param activation: the activation function of the node
        :param _id: the id of the node (same as the innovation of the respective node gene)
        """
        self.activation: Any = activations.functions[activation]
        self.id: int = _id

    def forward(self) -> float:
        """
        Returns the output of the node

        :return: The output of the node after passing through the activation function
        """
        return self.output if self.activation is None else self.activation(self.output)


class Network:
    def __init__(self, NodeGenes: tuple, LinkGenes: tuple, inputs: int, outputs) -> None:
        """
        The phenotype of an individual. This class creates the actual network that does the computations based
        on the genome provided. All the function arguments should be taken from the genome of a given individual.

        :param NodeGenes: A tuple containing all the node genes of the network.
        :param LinkGenes: A tuple containing all the link genes of the network.
        :param inputs: The number of input nodes
        :param outputs: The number of output nodes
        """
        self.inputs: int = inputs               # number of input nodes
        self.outputs: int = outputs             # number of output nodes
        self.nodes: list = [Node(gene.activation, gene.innovation) for gene in NodeGenes]
        self.links: list = [Link(gene.begin, gene.end, gene.weight) for gene in LinkGenes if gene.enabled]

        self.links: tuple = self.sort_links(self.links)

    def forward(self, inputs: tuple) -> list:
        """
        Takes a given input and returns a given output

        :return: The output of the network
        """

        for node in self.nodes:
            node.output = 0

        for i in range(0, len(inputs)):
            self.nodes[i].output = inputs[i]

        for link in self.links:
            self.nodes[link.end - 1].output += self.nodes[link.begin - 1].forward() * link.weight

        output_nodes: list = self.nodes[len(self.nodes) - self.outputs:len(self.nodes)]

        return [node.output for node in output_nodes]

    def sort_links(self, links: list) -> tuple:
        """
        Sorts the links in the order that they will transmit the signal. The technic used is similar to the
        technic explained in this video: https://www.youtube.com/watch?v=EvV5Qtp_fYg.
        A key difference is that this functions sorts the links instead of the nodes.

        :return: A tuple containing the links sorted in order to propagate the signal
        """

        left_nodes: list = deepcopy(self.nodes)
        left_links: list = deepcopy(links)
        sorted_links: tuple = tuple()

        while len(left_links) > 0:
            # creating a temporary list to remove nodes without unexpected behavior during the loop
            nodes_to_remove: list = list()

            for node in left_nodes:
                incoming_links: int = 0
                outgoing_links: list = list()

                for link in left_links:
                    if link.end == node.id:
                        incoming_links += 1
                    elif link.begin == node.id:
                        outgoing_links.append(link)

                if incoming_links == 0:
                    sorted_links = sorted_links + (*outgoing_links, )
                    left_links = [element for element in left_links if element not in outgoing_links]
                    nodes_to_remove.append(node)

            left_nodes = [element for element in left_nodes if element not in nodes_to_remove]

        return sorted_links

    def __str__(self) -> str:
        s: str = ""
        for link in self.links:
            s += f"{link.begin}------------{link.end}\n"

        return s
