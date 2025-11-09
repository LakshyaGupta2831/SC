# Initialize population Xi (i = 1 to N)
# Initialize a, A, C coefficients
# Evaluate fitness of each search agent
# Identify Alpha (best), Beta (2nd best), Delta (3rd best)

# While (t < max_iterations):
#     For each wolf i:
#         Update coefficients A, C
#         Compute Dα = |C1*Xα - Xi|
#         Compute Dβ = |C2*Xβ - Xi|
#         Compute Dδ = |C3*Xδ - Xi|
        
#         X1 = Xα - A1*Dα
#         X2 = Xβ - A2*Dβ
#         X3 = Xδ - A3*Dδ
        
#         Update position Xi = (X1 + X2 + X3) / 3
#     End For
    
#     Update a = 2 - (2 * t / max_iterations)
#     Evaluate all fitness values
#     Update Xα, Xβ, Xδ based on new best solutions
#     Increment t
# End While

# Return Xα as the best (optimal) solution



import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# Given Function 
def objective_function(x):
    x1, x2 = x[0], x[1]
    return x1**2 - x1*x2 + x2**2 + 2*x1 + 4*x2 + 3

def GWO(fitness_func, dim, lb, ub, N, Max_iter):
    X = np.random.uniform(lb, ub, (N, dim))
    fitness = np.apply_along_axis(fitness_func, 1, X)

    Alpha, Beta, Delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    fAlpha, fBeta, fDelta = float("inf"), float("inf"), float("inf")

    trajectories = np.zeros((N, Max_iter, dim))

    for i in range(N):
        if fitness[i] < fAlpha:
            fDelta, Delta = fBeta, Beta.copy()
            fBeta, Beta = fAlpha, Alpha.copy()
            fAlpha, Alpha = fitness[i], X[i].copy()
        elif fitness[i] < fBeta:
            fDelta, Delta = fBeta, Beta.copy()
            fBeta, Beta = fitness[i], X[i].copy()
        elif fitness[i] < fDelta:
            fDelta, Delta = fitness[i], X[i].copy()

    for t in range(Max_iter):
        a = 2 - 2 * (t / Max_iter)

        for i in range(N):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha[j] - X[i, j])
                X1 = Alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta[j] - X[i, j])
                X2 = Beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta[j] - X[i, j])
                X3 = Delta[j] - A3 * D_delta

                X[i, j] = (X1 + X2 + X3) / 3

            X[i] = np.clip(X[i], lb, ub)

        fitness = np.apply_along_axis(fitness_func, 1, X)
        trajectories[:, t, :] = X

        for i in range(N):
            if fitness[i] < fAlpha:
                fDelta, Delta = fBeta, Beta.copy()
                fBeta, Beta = fAlpha, Alpha.copy()
                fAlpha, Alpha = fitness[i], X[i].copy()
            elif fitness[i] < fBeta:
                fDelta, Delta = fBeta, Beta.copy()
                fBeta, Beta = fitness[i], X[i].copy()
            elif fitness[i] < fDelta:
                fDelta, Delta = fitness[i], X[i].copy()

        print(f"Iteration {t+1}/{Max_iter} --> Best Fitness: {fAlpha:.6f}")

    return Alpha, fAlpha, trajectories


# ---- Main Program ----
print("Grey Wolf Optimizer (3D Visualization)")

N = int(input("Enter number of wolves (N): "))
Max_iter = int(input("Enter maximum iterations (Max_iter): "))

dim = 2
lb, ub = -5, 5

best_pos, best_score, traj = GWO(objective_function, dim, lb, ub, N, Max_iter)

print("Optimization Completed!")
print("Best Position (x₁, x₂):", best_pos)
print("Best Fitness Value:", best_score)

# ---- 3D Visualization ----
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# x1 = np.linspace(lb, ub, 100)
# x2 = np.linspace(lb, ub, 100)
# X1, X2 = np.meshgrid(x1, x2)
# Z = X1**2 - X1*X2 + X2**2 + 2*X1 + 4*X2 + 3

# ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)

# ---- Initialize wolves ----
# wolf_lines = []
# colors = plt.cm.tab10(np.linspace(0, 1, N))

# for i in range(N):
#     (line,) = ax.plot([], [], [], 'o-', color=colors[i], label=f"Wolf {i}")
#     wolf_lines.append(line)

# # # Highlight best position
# # best_point = ax.scatter(best_pos[0], best_pos[1], best_score, c='r', s=80, label='Best Solution (α)')

# # ax.set_title("🐺 Grey Wolf Optimization Trajectories (3D View)")
# # ax.set_xlabel("x₁")
# # ax.set_ylabel("x₂")
# # ax.set_zlabel("f(x₁, x₂)")
# # ax.legend(loc="upper left", fontsize=8)
# # plt.tight_layout()

# def update(frame):
#     for i in range(N):
#         xs = traj[i, :frame, 0]
#         ys = traj[i, :frame, 1]
#         zs = [objective_function([xs[k], ys[k]]) for k in range(len(xs))]
#         wolf_lines[i].set_data(xs, ys)
#         wolf_lines[i].set_3d_properties(zs)
#     # ax.set_title(f" Grey Wolf Optimization (Iteration {frame+1}/{Max_iter})")
#     return wolf_lines

# ani = FuncAnimation(fig, update, frames=Max_iter, interval=600, blit=False, repeat=False)

# plt.show()
