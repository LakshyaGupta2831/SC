

import random

# --------- CROSSOVER FUNCTIONS ---------

# 1. Single Point Crossover
def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2, point

# 2. Two Point Crossover
def two_point_crossover(p1, p2):
    point1, point2 = sorted(random.sample(range(1, len(p1) - 1), 2))
    child1 = p1[:point1] + p2[point1:point2] + p1[point2:]
    child2 = p2[:point1] + p1[point1:point2] + p2[point2:]
    return child1, child2, (point1, point2)

# 3. Order Crossover (OX)
def order_crossover(p1, p2):
    start, end = sorted(random.sample(range(len(p1)), 2))
    child = [None] * len(p1)
    child[start:end] = p1[start:end]
    p2_items = [x for x in p2 if x not in child]
    j = 0
    for i in range(len(p1)):
        if child[i] is None:
            child[i] = p2_items[j]
            j += 1
    return child, (start, end)

# 4. Uniform Crossover
def uniform_crossover(p1, p2):
    mask = [random.randint(0, 1) for _ in range(len(p1))]
    child1 = [p1[i] if mask[i] else p2[i] for i in range(len(p1))]
    child2 = [p2[i] if mask[i] else p1[i] for i in range(len(p1))]
    return child1, child2, mask

# --------- MAIN PROGRAM ---------

# Input parents
print("Enter Parent 1 (comma-separated values, e.g. 1,2,3,4,5):")
parent1 = list(map(int, input().split(',')))

print("Enter Parent 2 (comma-separated values, same length):")
parent2 = list(map(int, input().split(',')))

if len(parent1) != len(parent2):
    print("Error: Parents must be of equal length!")
    exit()

# Menu
print("\nChoose a crossover type:")
print("1. Single Point")
print("2. Two Point")
print("3. Order (OX)")
print("4. Uniform")

choice = int(input("Enter your choice (1-4): "))

print("\nParent 1:", parent1)
print("Parent 2:", parent2)
print("-" * 50)

# Apply chosen crossover
if choice == 1:
    child1, child2, point = single_point_crossover(parent1, parent2)
    print(f"Single Point Crossover (point={point}):")
    print("Child 1:", child1)
    print("Child 2:", child2)

elif choice == 2:
    child1, child2, points = two_point_crossover(parent1, parent2)
    print(f"Two Point Crossover (points={points}):")
    print("Child 1:", child1)
    print("Child 2:", child2)

elif choice == 3:
    child, points = order_crossover(parent1, parent2)
    print(f"Order Crossover (segment={points}):")
    print("Child:", child)

elif choice == 4:
    child1, child2, mask = uniform_crossover(parent1, parent2)
    print(f"Uniform Crossover (mask={mask}):")
    print("Child 1:", child1)
    print("Child 2:", child2)

else:
    print("Invalid choice! Please enter 1–4.")
