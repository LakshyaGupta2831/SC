import random

print("=== 🏃 Rank-Based Selection ===")

# --- Input ---
choice = input("Enter data manually? (y/n): ").lower()
if choice == 'y':
    n = int(input("Enter number of individuals: "))
    names, fitness = [], []
    for i in range(n):
        names.append(input(f"Name {i+1}: "))
        fitness.append(float(input(f"Fitness of {names[-1]}: ")))
else:
    names = ["A", "B", "C", "D", "E"]
    fitness = [4, 6, 3, 7, 5]
    n = len(names)
    print("\nUsing default dataset...")

# --- Rank & Probability ---
data = sorted(zip(fitness, names))  # sort by fitness ascending
sorted_fit, sorted_names = zip(*data)
ranks = range(1, n + 1)
prob = [r / sum(ranks) for r in ranks]

# --- Display ---
print("\nIndividual | Fitness | Rank | Probability")
print("-" * 40)
for i in range(n):
    print(f"{sorted_names[i]:<10}{sorted_fit[i]:<9}{ranks[i]:<6}{prob[i]:.3f}")

# --- Selection ---
p = 0.4  # 40% selection rate
k = max(1, int(p * n))
selected = random.choices(sorted_names, weights=prob, k=k)

print(f"\nSelected ({k}): {selected}")
print("\n✅ Rank-Based Selection Completed.")
