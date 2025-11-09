import random

def steady_state_selection(pop, fit, pool_size, NU):
    mating_pool, log = [], []
    data = list(zip(pop, fit))
    i = 1

    while len(mating_pool) < pool_size:
        data.sort(key=lambda x: x[1], reverse=True)
        best = data[:NU]
        worst = data[-NU:]
        data = data[:-NU]  # remove worst

        for ind, _ in best:
            if len(mating_pool) < pool_size:
                mating_pool.append(ind)

        log.append((i, best, worst))
        i += 1

    return mating_pool, log

def main():
    print("=== 🧬 Steady-State Selection ===")
    print("1. Use example data\n2. Enter manually")
    ch = input("Enter choice: ")

    if ch == '1':
        pop = ["A", "B", "C", "D"]
        skill = [80, 70, 80, 85]
        exp = [4, 5, 2, 6]
        salary = [50000, 60000, 40000, 70000]
        fit = [(0.6 * skill[i]) + (0.3 * exp[i]) + (0.1 / (1 + salary[i])) for i in range(4)]
        print("\nUsing predefined dataset:")
        for i in range(4):
            print(f"{pop[i]}: Skill={skill[i]}, Exp={exp[i]}, Salary={salary[i]}, Fitness={fit[i]:.2f}")
    else:
        n = int(input("\nEnter number of individuals: "))
        pop, fit = [], []
        for i in range(n):
            name = input(f"Name {i+1}: ")
            s = float(input("Skill (0-100): "))
            e = float(input("Experience: "))
            sal = float(input("Salary: "))
            pop.append(name)
            fit.append((0.6 * s) + (0.3 * e) + (0.1 / (1 + sal)))

    pool_size = int(input("\nEnter mating pool size: "))
    NU = int(input("Enter NU (best/worst count per iteration): "))

    pool, log = steady_state_selection(pop, fit, pool_size, NU)

    print("\n=== Selection Log ===")
    for i, best, worst in log:
        print(f"\nIteration {i}:")
        print(" Selected:", [f"{n}({f:.2f})" for n, f in best])
        print(" Removed :", [f"{n}({f:.2f})" for n, f in worst])

    print("\n✅ Final Mating Pool:")
    for i, ind in enumerate(pool, 1):
        print(f"{i}. {ind}")

if __name__ == "__main__":
    main()
