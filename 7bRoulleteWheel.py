import numpy as np
import random
import matplotlib.pyplot as plt

def compute_fitness(speed, fuel, comfort):
    return 0.5 * speed + 0.3 * fuel + 0.2 * comfort

def roulette_selection(cars, fitness, n_select):
    prob = fitness / np.sum(fitness)
    cum_prob = np.cumsum(prob)
    selected = [cars[np.searchsorted(cum_prob, random.random())] for _ in range(n_select)]
    return selected, prob, cum_prob

def main():
    print("=== 🚗 Roulette Wheel Selection ===")
    print("1. Use predefined data\n2. Enter manually")
    choice = input("Enter choice: ")

    if choice == '1':
        cars = ['A', 'B', 'C', 'D']
        fitness = np.array([38.8, 36.7, 53.4, 58.4])
    else:
        n = int(input("Enter number of cars: "))
        cars, fitness = [], []
        for i in range(n):
            name = input(f"\nCar {i+1} name: ")
            s = float(input("Speed: "))
            f = float(input("Fuel Efficiency: "))
            c = float(input("Comfort: "))
            cars.append(name)
            fitness.append(compute_fitness(s, f, c))
        fitness = np.array(fitness)

    n_select = int(input("\nEnter number of cars to select: "))
    selected, prob, cum_prob = roulette_selection(cars, fitness, n_select)

    print("\nCar\tFitness\tProb\tCumProb")
    for i in range(len(cars)):
        print(f"{cars[i]}\t{fitness[i]:.1f}\t{prob[i]:.3f}\t{cum_prob[i]:.3f}")

    print("\n🎯 Selected Cars:", selected)

    # Plot fitness and probabilities
    plt.figure(figsize=(8, 4))
    plt.bar(cars, fitness, color='lightgreen', edgecolor='black')
    plt.title("Car Fitness Values")
    plt.xlabel("Cars"); plt.ylabel("Fitness")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()

    plt.pie(prob, labels=cars, autopct='%1.1f%%', startangle=90)
    plt.title("Selection Probability (Roulette Wheel)")
    plt.show()

if __name__ == "__main__":
    main()
