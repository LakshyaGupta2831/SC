# Union 

import matplotlib.pyplot as plt
import numpy as np

# Membership function for "Slow Speed" fuzzy set (Set A)
def mu_slow(speed):
    if speed <= 20:
        return 1.0
    elif 20 < speed <= 60:
        return (60.0 - speed) / 40.0
    else:
        return 0.0

# Membership function for "Fast Speed" fuzzy set (Set B)
def mu_fast(speed):
    if speed < 60:
        return 0.0
    elif 60 <= speed < 100:
        return (speed - 60.0) / 40.0
    else:
        return 1.0

# Main
if __name__ == "__main__":
    speeds = [20, 40, 60, 80, 100]

    # Print table
    print("Speed\tMF(A) (Slow)\tMF(B) (Fast)\tMF(Intersection)")
    for s in speeds:
        muA = mu_slow(s)
        muB = mu_fast(s)
        muIntersection = min(muA, muB)  # Intersection mai min
        print(f"{s}km/h\t{muA:.2f}\t\t{muB:.2f}\t\t{muIntersection:.2f}")

    # Generate smooth values for graph
    x = np.linspace(0, 120, 200)
    muA_vals = [mu_slow(i) for i in x]
    muB_vals = [mu_fast(i) for i in x]
    muIntersection_vals = [min(a, b) for a, b in zip(muA_vals, muB_vals)]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, muA_vals, label="Slow Speed (Set A)", linewidth=2)
    plt.plot(x, muB_vals, label="Fast Speed (Set B)", linewidth=2)
    plt.plot(x, muIntersection_vals, label="Intersection (A ∩ B)", linestyle="--", linewidth=2)

    plt.title("Fuzzy Intersection of Speed")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Membership Value")
    plt.legend()
    plt.grid(True)
    plt.show()



# Intersection 
def muLow(x):
    if x <= 20:
        return 1
    elif x <= 60:
        return (60-x) / 40
    else:
        return 0

def muHigh(x):
    if x < 40:
        return 0
    elif x <= 80:
        return (x - 40) / 40   # Gradually increases
    else:
        return 1

def getClass(a,b):
    if a < b : return 'Low'
    elif a > b : return 'High'
    else:
        return 'None'

No_of_cars = [10,30,50,60,70,30]

for c in No_of_cars:
    a = muLow(c)
    b = muHigh(c)
    print(c, round(a,2),round(b,2),round(min(a,b),2), getClass(a,b),sep='\t')

import matplotlib.pyplot as plt

x = range(0,101)
plt.plot(x, [muLow(i) for i in x])
plt.plot(x, [muHigh(i) for i in x])
plt.plot(x, [min(muLow(i), muHigh(i)) for i in x], '--')
plt.show()



# Compliment
import numpy as np
import matplotlib.pyplot as plt

# Define blood sugar range (mg/dL)
x = np.linspace(70, 200, 300)

# --- Fuzzy set: "High Blood Sugar" ---
# Using a smooth membership function (sigmoid)
def high_blood_sugar(x):
    return 1 / (1 + np.exp(-0.1 * (x - 130)))  # midpoint ~130 mg/dL

# --- Complement fuzzy set: "Not High Blood Sugar" ---
def not_high_blood_sugar(x):
    return 1 - high_blood_sugar(x)

# Calculate memberships
mu_high = high_blood_sugar(x)
mu_not_high = not_high_blood_sugar(x)

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(x, mu_high, label="High Blood Sugar (A)", color='red', linewidth=2)
plt.plot(x, mu_not_high, label="Not High Blood Sugar (A')", color='green', linewidth=2, linestyle='--')

plt.title("Fuzzy Complement Operation: Medical Diagnosis System", fontsize=12)
plt.xlabel("Blood Sugar Level (mg/dL)")
plt.ylabel("Membership Value (μ)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
