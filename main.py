import gene
import network
from neat import Individual

gene.ConnectInputs = True
gene.BiasNeurons = True

individual1 = Individual(3, 2)

print(individual1.genome)
print(individual1.network)

for c in range(0, 100):
    individual1.genome.mutate()


print(individual1.genome)
individual1.network.__init__(individual1.genome.NodeGenes, individual1.genome.LinkGenes, individual1.genome.inputs, individual1.genome.outputs)
print(individual1.network)