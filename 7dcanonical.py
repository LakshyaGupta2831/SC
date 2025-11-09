import random

print("=== Canonical Selection ===")
print("1. Use example data")
print("2. Enter manually")
choice = int(input("Enter choice: "))

# Input data
if choice == 1:
    names = ["A", "B", "C", "D", "E"]
    fi = [4, 6, 3, 7, 5]
else:
    n = int(input("Enter number of individuals: "))
    names, fi = [], []
    for i in range(n):
        names.append(input(f"Name {i+1}: "))
        fi.append(float(input("Fitness value: ")))

# Fitness calculation
F_bar = sum(fi) / len(fi)
fitness = [round(f / F_bar, 2) for f in fi]

# Selection
p = float(input("Enter selection rate (e.g. 0.4 for 40%): "))
k = max(1, int(p * len(names)))
selected = random.choices(names, weights=fitness, k=k)

# Output
print("\nIndividuals:", names)
print("fi:", fi)
print("F̄ =", round(F_bar, 2))
print("Fitness:", fitness)
print(f"\nSelected ({k}):", selected)
print("\n✅ Process Completed")
