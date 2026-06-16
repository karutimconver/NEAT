import gene
import network
from neat import Individual, NEAT

neat = NEAT(3, 2, 2, 1)
print(neat.population)

penis = neat.crossover(*neat.population)

print(penis[0].genome)
print(penis[1].genome)