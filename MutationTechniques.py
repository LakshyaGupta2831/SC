import random

# --- Define sample chromosomes ---
binary_chrom = [1, 0, 0, 1, 1, 0, 1]       # for flipping & reversing
order_chrom = ['A', 'B', 'C', 'D', 'E']    # for inversion & swap


# --- 1️⃣ Flipping Mutation (Binary GA) ---
def flip_mutation(chrom, rate=0.3):
    new = chrom.copy()
    for i in range(len(chrom)):
        if random.random() < rate:
            new[i] = 1 - new[i]  # Flip 0→1 or 1→0
    return new


# --- 2️⃣ Reversing Mutation (Binary GA) ---
def reverse_mutation(chrom):
    new = chrom.copy()
    i, j = sorted(random.sample(range(len(chrom)), 2))
    new[i:j] = reversed(new[i:j])  # Reverse a segment
    return new


# --- 3️⃣ Inversion Mutation (Order GA / TSP) ---
def inversion_mutation(chrom):
    new = chrom.copy()
    i, j = sorted(random.sample(range(len(chrom)), 2))
    new[i:j] = reversed(new[i:j])  # Same concept as reverse, applied to ordered genes
    return new


# --- 4️⃣ Swap Mutation (Order GA / TSP) ---
def swap_mutation(chrom):
    new = chrom.copy()
    i, j = random.sample(range(len(chrom)), 2)
    new[i], new[j] = new[j], new[i]  # Swap two positions
    return new


# --- MAIN EXECUTION ---
def main():
    print("=== 🧬 Mutation Techniques Demonstration ===")

    print("\nOriginal Binary Chromosome:", binary_chrom)
    print("Flipping Mutation:", flip_mutation(binary_chrom))
    print("Reversing Mutation:", reverse_mutation(binary_chrom))

    print("\nOriginal Order Chromosome:", order_chrom)
    print("Inversion Mutation:", inversion_mutation(order_chrom))
    print("Swap Mutation:", swap_mutation(order_chrom))

    print("\n✅ Demonstration Complete")

if __name__ == "__main__":
    main()
