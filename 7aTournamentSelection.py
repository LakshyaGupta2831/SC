import random

def fitness_calc(skill, exp, salary):
    return (0.6 * skill) + (0.3 * exp) + (0.1 / (1 + salary))

def tournament_selection(pop, fit, pool_size, t_size):
    mating_pool = []
    print("\nTrial | Contestants | Winner (Fitness)")
    print("-" * 45)
    for trial in range(1, pool_size + 1):
        contenders = random.sample(range(len(pop)), t_size)
        best = max(contenders, key=lambda i: fit[i])
        mating_pool.append(pop[best])
        names = ", ".join(pop[i] for i in contenders)
        print(f"{trial:3d}   | {names:<20} | {pop[best]} ({fit[best]:.2f})")
    return mating_pool

def main():
    print("=== 🧬 Tournament Selection ===")
    print("1. Use example data\n2. Enter manually")
    ch = input("Enter choice: ")

    if ch == '1':
        pop = ["A", "B", "C", "D"]
        skill = [80, 70, 80, 85]
        exp = [4, 5, 2, 6]
        salary = [50000, 60000, 40000, 70000]
        fit = [fitness_calc(skill[i], exp[i], salary[i]) for i in range(4)]
    else:
        n = int(input("Enter number of individuals: "))
        pop, fit = [], []
        for i in range(n):
            name = input(f"\nName {i+1}: ")
            s = float(input("Skill (0-100): "))
            e = float(input("Experience (yrs): "))
            sal = float(input("Salary expectation: "))
            pop.append(name)
            fit.append(fitness_calc(s, e, sal))

    Np = int(input("\nEnter mating pool size: "))
    Nt = int(input("Enter tournament size: "))

    selected = tournament_selection(pop, fit, Np, Nt)

    print("\n✅ Final Selected Mating Pool:")
    for i, s in enumerate(selected, 1):
        print(f"{i}. {s}")

if __name__ == "__main__":
    main()
