import numpy as np
import matplotlib.pyplot as plt

# ----- Membership Functions -----
def trimf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def trapmf(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a), 1), (d - x)/(d - c)), 0)

def gaussmf(x, c, sigma):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

def sigmf(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))

def gbellmf(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a)**(2*b))

# ----- Dynamic Input -----
print("Choose the Membership Function:")
print("1. Triangular (a, b, c)")
print("2. Trapezoidal (a, b, c, d)")
print("3. Gaussian (c, sigma)")
print("4. Sigmoidal (a, c)")
print("5. Bell-shaped (a, b, c)")

choice = input("\nEnter choice (1-5): ").strip()

x = np.linspace(0, 10, 1000)

# ----- Select Function Based on User Choice -----
if choice == '1':
    a, b, c = map(float, input("Enter a, b, c (space separated): ").split())
    y = trimf(x, a, b, c)
    title = "Triangular Membership Function"

elif choice == '2':
    a, b, c, d = map(float, input("Enter a, b, c, d (space separated): ").split())
    y = trapmf(x, a, b, c, d)
    title = "Trapezoidal Membership Function"

elif choice == '3':
    c, sigma = map(float, input("Enter c, sigma (space separated): ").split())
    y = gaussmf(x, c, sigma)
    title = "Gaussian Membership Function"

elif choice == '4':
    a, c = map(float, input("Enter a, c (space separated): ").split())
    y = sigmf(x, a, c)
    title = "Sigmoidal Membership Function"

elif choice == '5':
    a, b, c = map(float, input("Enter a, b, c (space separated): ").split())
    y = gbellmf(x, a, b, c)
    title = "Bell-shaped Membership Function"

else:
    print("Invalid choice. Exiting.")
    exit()

# ----- Plot the Selected Function -----
plt.figure(figsize=(7,4))
plt.plot(x, y, linewidth=2, color='teal')
plt.title(title)
plt.xlabel("x")
plt.ylabel("μ(x)")
plt.grid(True)
plt.show()
