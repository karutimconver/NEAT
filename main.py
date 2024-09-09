import gene
import network
from neat import Individual

gene.ConnectInputs = True
gene.BiasNeurons = True

individual1 = Individual(3, 2)

print(individual1.genome)
print(individual1.network)
