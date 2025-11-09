import random
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define cities and distances ---
cities = ['A', 'B', 'C', 'D', 'E']
distance_matrix = {
    ('A','B'):28, ('A','C'):6,  ('A','D'):23, ('A','E'):22,
    ('B','C'):12, ('B','D'):12, ('B','E'):16,
    ('C','D'):29, ('C','E'):4,
    ('D','E'):15
}

def get_distance(a, b):
    if (a, b) in distance_matrix: return distance_matrix[(a, b)]
    if (b, a) in distance_matrix: return distance_matrix[(b, a)]
    return float('inf')

def route_distance(route):
    return sum(get_distance(route[i], route[i+1]) for i in range(len(route)-1)) + get_distance(route[-1], route[0])

# --- Step 2: GA components ---
def initialize_population(pop_size):
    return [random.sample(cities, len(cities)) for _ in range(pop_size)]

def crossover(p1, p2):
    start, end = sorted(random.sample(range(len(p1)), 2))
    child = [None]*len(p1)
    child[start:end] = p1[start:end]
    pointer = 0
    for c in p2:
        if c not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = c
    return child

def mutate(route, rate):
    new = route.copy()
    if random.random() < rate:
        i, j = random.sample(range(len(route)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def select(pop, fitness):
    idx = np.argsort(fitness)
    return [pop[i] for i in idx[:len(pop)//2]]

# --- Step 3: Genetic Algorithm ---
def genetic_algorithm(generations, pop_size, mutation_rate):
    pop = initialize_population(pop_size)
    best_dist_list = []

    for g in range(generations):
        fitness = [route_distance(r) for r in pop]
        best_idx = np.argmin(fitness)
        best_route, best_dist = pop[best_idx], fitness[best_idx]
        best_dist_list.append(best_dist)

        selected = select(pop, fitness)
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = mutate(crossover(p1, p2), mutation_rate)
            children.append(child)
        pop = children

        print(f"Gen {g+1}: Best Distance = {best_dist:.2f}, Route = {'-'.join(best_route)}")

    return best_route, best_dist_list

# --- Step 4: Run program ---
def main():
    print("=== 🧬 Travelling Salesman Problem using Genetic Algorithm ===")
    generations = int(input("Enter number of generations (e.g. 30): "))
    pop_size = int(input("Enter population size (e.g. 10): "))
    mutation_rate = float(input("Enter mutation rate (0.0 - 1.0): "))

    best_route, best_dists = genetic_algorithm(generations, pop_size, mutation_rate)

    print("\n✅ Best Route Found:", " → ".join(best_route))
    print("Total Distance:", route_distance(best_route))

    # Plot distance improvement
    plt.plot(range(1, generations+1), best_dists, marker='o')
    plt.title("Distance Improvement Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
