import random

# function to generate a random population
def generate_population(size):
	population = []
	for _ in range(size):
		genes = [0, 1]
		chromosome = []
		for _ in range(len(items)):
			chromosome.append(random.choice(genes))
		population.append(chromosome)
	print("Generated a random population of size", size)
	return population

# function to calculate the fitness of a chromosome
def calculate_fitness(chromosome):
	total_weight = 0
	total_value = 0
	for i in range(len(chromosome)):
		if chromosome[i] == 1:
			total_weight += items[i][0]
			total_value += items[i][1]
	if total_weight > max_weight:
		return 0
	else:
		return total_value

# function to select two chromosomes for crossover
def select_chromosomes(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))

	fitness_values = [float(i)/sum(fitness_values) for i in fitness_values]

	parent1 = random.choices(population, weights=fitness_values, k=1)[0]
	parent2 = random.choices(population, weights=fitness_values, k=1)[0]

	print("Selected two chromosomes for crossover")
	return parent1, parent2

# function to perform crossover between two chromosomes
def crossover(parent1, parent2):
	crossover_point = random.randint(0, len(items)-1)
	child1 = parent1[0:crossover_point] + parent2[crossover_point:]
	child2 = parent2[0:crossover_point] + parent1[crossover_point:]

	print("Performed crossover between two chromosomes")
	return child1, child2

# function to perform mutation on a chromosome
def mutate(chromosome):
	mutation_point = random.randint(0, len(items)-1)
	if chromosome[mutation_point] == 0:
		chromosome[mutation_point] = 1
	else:
		chromosome[mutation_point] = 0
	print("Performed mutation on a chromosome")
	return chromosome

# function to get the best chromosome from the population
def get_best(population):
	fitness_values = []
	for chromosome in population:
		fitness_values.append(calculate_fitness(chromosome))

	max_value = max(fitness_values)
	max_index = fitness_values.index(max_value)
	return population[max_index]


# items that can be put in the knapsack
items = [# [weight, value]
		[2, 3],
		[3, 4],
		[4, 5],
		[5, 8],
		[6, 9]
	]

# print available items
print("Available items:\n", items)

# parameters for genetic algorithm
max_weight = 10
population_size = 10
mutation_probability = 0.2
generations = 10

print("\nGenetic algorithm parameters:")
print("Max weight:", max_weight)
print("Population:", population_size)
print("Mutation probability:", mutation_probability)
print("Generations:", generations, "\n")
print("Performing genetic evolution:")

# generate a random population
population = generate_population(population_size)

# evolve the population for specified number of generations
for _ in range(generations):
	# select two chromosomes for crossover
	parent1, parent2 = select_chromosomes(population)

	# perform crossover to generate two new chromosomes
	child1, child2 = crossover(parent1, parent2)

	# perform mutation on the two new chromosomes
	if random.uniform(0, 1) < mutation_probability:
		child1 = mutate(child1)
	if random.uniform(0, 1) < mutation_probability:
		child2 = mutate(child2)

	# replace the old population with the new population
	population = [child1, child2] + population[2:]

# get the best chromosome from the population
best = get_best(population)

# get the weight and value of the best solution
total_weight = 0
total_value = 0
for i in range(len(best)):
	if best[i] == 1:
		total_weight += items[i][0]
		total_value += items[i][1]

# print the best solution
print("\nThe best solution:")
print("Weight:", total_weight)
print("Value:", total_value)
