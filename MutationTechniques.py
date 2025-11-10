import random

# --- 1️⃣ Flipping Mutation (for Binary Chromosome) ---
def flip_mutation(chrom, rate=0.3):
    new = chrom.copy()
    for i in range(len(chrom)):
        if random.random() < rate:
            new[i] = 1 - new[i]  # Flip 0 → 1 or 1 → 0
    return new

# --- 2️⃣ Reversing Mutation (for Binary Chromosome) ---
def reverse_mutation(chrom):
    new = chrom.copy()
    i, j = sorted(random.sample(range(len(chrom)), 2))
    new[i:j] = reversed(new[i:j])  # Reverse a segment between i and j
    return new

# --- 3️⃣ Inversion Mutation (for Order-based Chromosome / TSP) ---
def inversion_mutation(chrom):
    new = chrom.copy()
    i, j = sorted(random.sample(range(len(chrom)), 2))
    new[i:j] = reversed(new[i:j])  # Reverse a segment in order chromosome
    return new

# --- 4️⃣ Swap Mutation (for Order-based Chromosome / TSP) ---
def swap_mutation(chrom):
    new = chrom.copy()
    i, j = random.sample(range(len(chrom)), 2)
    new[i], new[j] = new[j], new[i]  # Swap two random positions
    return new


# --- MAIN PROGRAM ---
def main():
    print("=== 🧬 Mutation Techniques (Dynamic Version) ===\n")

    print("Choose chromosome type:")
    print("1. Binary (e.g. 1,0,0,1,1)")
    print("2. Order-based (e.g. A,B,C,D,E)")
    ch_type = int(input("Enter your choice (1 or 2): "))

    if ch_type == 1:
        chrom = list(map(int, input("Enter Binary Chromosome (comma-separated 0s and 1s): ").split(',')))
    elif ch_type == 2:
        chrom = input("Enter Order Chromosome (comma-separated letters or symbols): ").split(',')
    else:
        print("Invalid choice!")
        return

    print("\nAvailable Mutations:")
    if ch_type == 1:
        print("1. Flipping Mutation")
        print("2. Reversing Mutation")
        choice = int(input("Enter mutation choice (1-2): "))

        if choice == 1:
            rate = float(input("Enter mutation rate (0-1): "))
            new_chrom = flip_mutation(chrom, rate)
            print(f"\nOriginal Chromosome: {chrom}")
            print(f"After Flipping Mutation: {new_chrom}")

        elif choice == 2:
            new_chrom = reverse_mutation(chrom)
            print(f"\nOriginal Chromosome: {chrom}")
            print(f"After Reversing Mutation: {new_chrom}")

        else:
            print("Invalid choice!")

    elif ch_type == 2:
        print("1. Inversion Mutation")
        print("2. Swap Mutation")
        choice = int(input("Enter mutation choice (1-2): "))

        if choice == 1:
            new_chrom = inversion_mutation(chrom)
            print(f"\nOriginal Chromosome: {chrom}")
            print(f"After Inversion Mutation: {new_chrom}")

        elif choice == 2:
            new_chrom = swap_mutation(chrom)
            print(f"\nOriginal Chromosome: {chrom}")
            print(f"After Swap Mutation: {new_chrom}")

        else:
            print("Invalid choice!")

    print("\n✅ Mutation Demonstration Complete.")


if __name__ == "__main__":
    main()
