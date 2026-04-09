import random

class Species:
    
    def __init__(self, mascot):

        self.mascot = mascot
        self.members = list()
        self.members.append(mascot)
        self.fitness_pop = list()
        self.total_adjusted_fitness = 0
    

    def add_adjusted_fitness(self, adjusted_fitness):
        self.total_adjusted_fitness += adjusted_fitness
    

    def reset(self):
        self.mascot = random.choice(self.members)
        self.members.clear()
        self.fitness_pop.clear()
        self.total_adjusted_fitness = 0