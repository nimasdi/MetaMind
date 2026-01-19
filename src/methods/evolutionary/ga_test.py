from ga import GeneticAlgorithm


class TSPProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.size = len(distance_matrix)

    def evaluate(self, individual):
        total = 0
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]
            total += self.distance_matrix[from_city][to_city]
        return total
    
    def __len__(self):
        return self.size

distance_matrix = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]

problem = TSPProblem(distance_matrix)

ga = GeneticAlgorithm(
    population_size=100,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    selection="tournament",
    tournament_size=3,
    elitism=2,
    crossover_type="pmx"
)

results = ga.fit(problem, problem_size=4)

print("Best tour:", results['best_individual'])
print("Best fitness (total distance):", results['best_fitness'])