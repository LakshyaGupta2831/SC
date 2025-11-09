"""
===============================================
   PARTICLE SWARM OPTIMIZATION (PSO) Algorithm
   Example: Minimizing the Rosenbrock Function
===============================================

# --- Algorithm Steps ---
1. Initialize a swarm of particles with:
   - Random positions p[i] within bounds.
   - Random velocities v[i].
2. Evaluate fitness f[i] for each particle using the objective function (Rosenbrock).
3. Set each particle’s best-known position (pbest) to its initial position.
4. Identify the global best particle (gbest) having the lowest fitness.
5. Repeat for a fixed number of iterations:
   a. For each particle:
      - Update velocity:
        v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
      - Update position:
        pos = pos + v
      - Clip positions within bounds.
   b. Evaluate new fitness values.
   c. Update personal best (pbest) and global best (gbest).
6. Return the best position (solution) and its fitness value.

# --- Pseudocode ---
Initialize w, c1, c2, bounds, and number of particles N
For each particle i:
    Randomly initialize position p[i] and velocity v[i]
    Evaluate fitness f[i] = F(p[i])
    pbest[i] = p[i]
Set gbest = best(pbest)

For iteration t = 1 to max_iter:
    For each particle i:
        v[i] = w*v[i] + c1*r1*(pbest[i]-p[i]) + c2*r2*(gbest-p[i])
        p[i] = p[i] + v[i]
        Keep p[i] within bounds
        Evaluate fitness f[i]
        If f[i] < pbest_fitness[i]:
            pbest[i] = p[i]
    Update gbest as best of pbest
Output gbest and its fitness
"""

import numpy as np
# import matplotlib.pyplot as plt
# import os

DIMENSIONS = 2  
W = 0.9         
C1 = 1.5        
C2 = 1.5        
BOUNDS = (-5, 5)

np.random.seed(42)

# ---------------- Rosenbrock Function ---------------- #
def rosenbrock_function(x):
    """
    Generalized Rosenbrock function for n-dimensional input.
    f(x) = Σ_{i=1}^{n-1} [100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
    Minimum is 0 at x = [1, 1, ..., 1]
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def initialize_swarm(num_particles, dimensions):
    """Initialize particle positions and velocities randomly"""
    positions = np.random.uniform(BOUNDS[0], BOUNDS[1], (num_particles, dimensions))
    velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
    return positions, velocities


def evaluate_fitness(positions):
    """Evaluate Rosenbrock fitness for all particles"""
    return np.array([rosenbrock_function(pos) for pos in positions])


def update_best_positions(positions, fitness, pbest_pos, pbest_fit, gbest_pos, gbest_fit):
    """Update personal and global bests"""
    improved = fitness < pbest_fit
    pbest_pos[improved] = positions[improved]
    pbest_fit[improved] = fitness[improved]
    
    best_idx = np.argmin(pbest_fit)
    if pbest_fit[best_idx] < gbest_fit:
        gbest_fit = pbest_fit[best_idx]
        gbest_pos = pbest_pos[best_idx].copy()
    
    return pbest_pos, pbest_fit, gbest_pos, gbest_fit


def update_velocity_and_position(positions, velocities, pbest_pos, gbest_pos):
    """Standard PSO update rule"""
    r1 = np.random.rand(*positions.shape)
    r2 = np.random.rand(*positions.shape)
    
    cognitive = C1 * r1 * (pbest_pos - positions)
    social = C2 * r2 * (gbest_pos - positions)
    velocities = W * velocities + cognitive + social
    
    positions = positions + velocities
    positions = np.clip(positions, BOUNDS[0], BOUNDS[1])
    
    return positions, velocities


def run_pso(num_particles, num_iterations):
    """Main PSO loop"""
    positions, velocities = initialize_swarm(num_particles, DIMENSIONS)
    pbest_pos = positions.copy()
    pbest_fit = evaluate_fitness(positions)
    gbest_pos = pbest_pos[np.argmin(pbest_fit)].copy()
    gbest_fit = np.min(pbest_fit)
    
    history = [positions.copy()]
    print(f"Initial best: {gbest_fit:.6f} at {gbest_pos}")
    
    for iteration in range(num_iterations):
        fitness = evaluate_fitness(positions)
        pbest_pos, pbest_fit, gbest_pos, gbest_fit = update_best_positions(
            positions, fitness, pbest_pos, pbest_fit, gbest_pos, gbest_fit
        )
        positions, velocities = update_velocity_and_position(
            positions, velocities, pbest_pos, gbest_pos
        )
        
        history.append(positions.copy())
        print(f"Iteration {iteration+1}: Best = {gbest_fit:.6f} at {gbest_pos}")
    
    return np.array(history), gbest_pos, gbest_fit


# def plot_results(history, num_particles):
#     """2D plot for visualization (only if DIMENSIONS = 2)"""
#     if DIMENSIONS != 2:
#         print("Skipping plot — visualization only supported for 2D Rosenbrock.")
#         return
    
#     plt.figure(figsize=(10, 6))
#     for i in range(num_particles):
#         traj = history[:, i, :]
#         plt.plot(traj[:, 0], traj[:, 1], 'o-', label=f'Particle {i}', alpha=0.7)
#         plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10)  # start
#         plt.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15)  # end
    
#     plt.xlabel('X₁')
#     plt.ylabel('X₂')
#     plt.title('PSO on Rosenbrock Function')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     os.makedirs('graphs', exist_ok=True)
#     plt.savefig('graphs/pso_rosenbrock.png', dpi=150, bbox_inches='tight')
#     plt.show()


if __name__ == "__main__":
    print("=== Particle Swarm Optimization on Rosenbrock Function ===")
    
    # Dynamic user input
    try:
        NUM_PARTICLES = int(input("Enter number of particles: "))
        NUM_ITERATIONS = int(input("Enter number of iterations: "))
    except ValueError:
        print("Invalid input! Using default values (Particles=5, Iterations=10).")
        NUM_PARTICLES, NUM_ITERATIONS = 5, 10

    print(f"\nParticles: {NUM_PARTICLES}, Iterations: {NUM_ITERATIONS}, Dimensions: {DIMENSIONS}\n")
    
    history, best_pos, best_fit = run_pso(NUM_PARTICLES, NUM_ITERATIONS)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULT:")
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fit:.10f}")
    print(f"Optimal solution: [1, 1, ..., 1] → fitness = 0")
    print(f"{'='*50}")
    
    # plot_results(history, NUM_PARTICLES)

