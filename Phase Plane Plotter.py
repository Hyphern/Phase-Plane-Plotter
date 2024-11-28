import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of equations: dx/dt = f(x, y), dy/dt = g(x, y)
def system(t, z):
    x, y = z
    dxdt = -1*x + 6*y
    dydt = -3*x + 8*y
    return [dxdt, dydt]

# Add trajectories
initial_conditions = [(-1, 0), (1, 0), (0, 1), (0, -1)] # Adjustable
t_bound = 40
t_span = [0, t_bound]  # Time range for integration

# Create a grid of points in the phase space
bound = 10
x = np.linspace(-bound, bound, 30) # Adjustable
y = np.linspace(-bound, bound, 30)
X, Y = np.meshgrid(x, y)

# Compute the vector field
U = -1*X + 6*Y # dx/dt (mimics the system of equations)
V = -3*X + 8*Y # dy/dt
X = np.clip(X, -10, 10)
Y = np.clip(Y, -10, 10)
magnitude = np.sqrt(U**2 + V**2)  # Normalize vectors for better visualization
U /= magnitude
V /= magnitude

# Everything below this point is for plotting
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='blue')

for x0, y0 in initial_conditions:
    # Forward integration
    sol_forward = solve_ivp(system, t_span, [x0, y0], t_eval=np.linspace(0, t_bound, 300))

    plt.plot(sol_forward.y[0], sol_forward.y[1], 'r')
    
    # Backward integration
    sol_backward = solve_ivp(system, [-t_span[1], 0], [x0, y0], t_eval=np.linspace(-t_bound, 0, 300))
    plt.plot(sol_backward.y[0], sol_backward.y[1], 'r')


def plot_eigenvectors(matrix, ax=None, range_limit=10):

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Eigenvectors
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]
    
    # Set up the plot if no axis is provided
    if ax is None:
        ax = plt.gca()
    
    # Define the range for plotting (-range_limit to range_limit)
    x_vals = np.array([-range_limit, range_limit])
    
    # Calculate the corresponding y-values for each eigenvector
    y_vals_v1 = (v1[1] / v1[0]) * x_vals  # Slope * x_vals for eigenvector 1
    y_vals_v2 = (v2[1] / v2[0]) * x_vals  # Slope * x_vals for eigenvector 2
    
    # Plot eigenvectors as lines in the specified range
    ax.plot(x_vals, y_vals_v1, color='green', label=f"Eigenvector 1 (λ = {eigenvalues[0]:.2f})")
    ax.plot(x_vals, y_vals_v2, color='magenta', label=f"Eigenvector 2 (λ = {eigenvalues[1]:.2f})")
    
    # Labeling
    ax.set_xlim(-range_limit, range_limit)
    ax.set_ylim(-range_limit, range_limit)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Eigenvectors as Lines in Range [-10, 10]')
    ax.grid(True)
    ax.legend()

# Define the system matrix (from the system of equations)
A = np.array([[-1, 6], [-3, 8]])

plot_eigenvectors(A)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Phase Portrait')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(alpha=0.3)
plt.axis('equal')
plt.xlim(-bound, bound)
plt.ylim(-bound, bound)
plt.show()
