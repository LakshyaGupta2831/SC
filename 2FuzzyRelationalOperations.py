import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Define Fuzzy Relations
# ----------------------------

# Relation R: Temperature (A) → Soil Moisture (B)
# Rows: Low, Medium, High temperature
# Columns: Dry, Normal, Wet moisture
R = np.array([
    [0.9, 0.7, 0.2],  # Low temp → mostly dry/normal
    [0.4, 0.8, 0.6],  # Medium temp → balanced
    [0.1, 0.5, 0.9]   # High temp → tends to wet
])

# Relation S: Soil Moisture (B) → Crop Suitability (C)
# Rows: Dry, Normal, Wet
# Columns: Wheat, Rice, Maize
S = np.array([
    [0.8, 0.2, 0.6],  # Dry soil best for Wheat/Maize
    [0.7, 0.5, 0.7],  # Normal soil moderate for all
    [0.3, 0.9, 0.5]   # Wet soil best for Rice
])

# ----------------------------
# Step 2: Max–Min Composition (T = R ○ S)
# ----------------------------

rows_R, cols_R = R.shape
rows_S, cols_S = S.shape

T = np.zeros((rows_R, cols_S))

for i in range(rows_R):
    for j in range(cols_S):
        mins = [min(R[i][k], S[k][j]) for k in range(cols_R)]
        T[i][j] = max(mins)

# ----------------------------
# Step 3: Display Results
# ----------------------------
print("Fuzzy Relation R (Temperature → Moisture):")
print(R)
print("\nFuzzy Relation S (Moisture → Crop):")
print(S)
print("\nResultant Relation T = R ○ S (Temperature → Crop):")
print(T)

labels_A = ["Low Temp", "Medium Temp", "High Temp"]
labels_C = ["Wheat", "Rice", "Maize"]

print("\nInterpretation (Crop suitability based on temperature):")
for i in range(len(labels_A)):
    print(f"{labels_A[i]} → Wheat={T[i][0]:.2f}, Rice={T[i][1]:.2f}, Maize={T[i][2]:.2f}")

# ----------------------------
# Step 4: Graph Visualization
# ----------------------------

x = np.arange(len(labels_A))  # Temperature levels (0, 1, 2)

plt.figure(figsize=(8, 5))
plt.plot(x, T[:, 0], marker='o', linewidth=2, label="Wheat")
plt.plot(x, T[:, 1], marker='o', linewidth=2, label="Rice")
plt.plot(x, T[:, 2], marker='o', linewidth=2, label="Maize")

plt.xticks(x, labels_A)
plt.xlabel("Temperature Levels")
plt.ylabel("Crop Suitability (Membership Value)")
plt.title("Fuzzy Relational Composition: Temperature → Crop Suitability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
