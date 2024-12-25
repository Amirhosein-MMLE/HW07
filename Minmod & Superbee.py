import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
x_min, x_max = -5.0, 5.0  # Domain
N = 1000  # Number of grid points
dx = (x_max - x_min) / N  # Grid spacing
x = np.linspace(x_min + dx / 2, x_max - dx / 2, N)  # Cell centers
dt = 0.01  # Time step
T = 2.0  # Final time
num_steps = int(T / dt)

# Initial condition
u_initial = np.zeros(N, dtype=np.float64)
u_initial[(x >= -1) & (x < 0)] = -1.0
u_initial[(x >= 0) & (x < 1)] = 1.0

# Define the Minmod limiter
def minmod(r):
    return np.maximum(0, np.minimum(1, r))

# Define the Superbee limiter
def superbee(r):
    return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))

# Flux function
def flux(u):
    return 0.5 * u**2

# Compute the piecewise linear slopes
def compute_slopes(u, limiter="minmod"):
    # Calculate the differences
    du_plus = u[2:] - u[1:-1]
    du_minus = u[1:-1] - u[:-2]
    
    # Ratio of differences
    r = np.zeros_like(du_minus)
    mask = du_minus != 0
    r[mask] = du_plus[mask] / (du_minus[mask] + 1e-12)  # Avoid division by zero
    
    # Apply the chosen limiter
    if limiter == "minmod":
        phi = minmod(r)
    elif limiter == "superbee":
        phi = superbee(r)
    else:
        raise ValueError("Unknown limiter: {}".format(limiter))
    
    return phi * du_minus

# Numerical flux functions
def godunov_flux(u_left, u_right):
    f_left = flux(u_left)
    f_right = flux(u_right)
    return np.where(u_left + u_right > 0, f_left, f_right)

def roe_flux(u_left, u_right):
    # Roe average for the nonlinear flux
    f_left = flux(u_left)
    f_right = flux(u_right)
    roe_avg = (u_left + u_right) / 2
    return 0.5 * (f_left + f_right - np.abs(roe_avg) * (u_right - u_left))

def lax_friedrichs_flux(u_left, u_right, alpha):
    f_left = flux(u_left)
    f_right = flux(u_right)
    return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

# Solver function for each method
def solve(method="godunov", limiter="minmod"):
    u = u_initial.copy()
    for n in range(num_steps):
        # Extend u to ghost cells
        u_ext = np.zeros(N + 2, dtype=np.float64)
        u_ext[1:-1] = u
        u_ext[0] = u[0]  # Left ghost cell (copy boundary)
        u_ext[-1] = u[-1]  # Right ghost cell (copy boundary)

        # Compute slopes
        slopes = compute_slopes(u_ext, limiter=limiter)
        slopes = np.concatenate(([0], slopes, [0]))  # Extend slopes to ghost cells

        # Compute left and right states at cell interfaces
        u_left = u_ext[:-1] + 0.5 * slopes[:-1]
        u_right = u_ext[1:] - 0.5 * slopes[1:]

        # Compute numerical fluxes
        if method == "godunov":
            num_flux = godunov_flux(u_left, u_right)
        elif method == "roe":
            num_flux = roe_flux(u_left, u_right)
        elif method == "lax_friedrichs":
            alpha = np.max(np.abs(u_ext))  # Lax-Friedrichs constant
            num_flux = lax_friedrichs_flux(u_left, u_right, alpha)
        else:
            raise ValueError("Unknown method: {}".format(method))

        # Update u using the conservative form
        u -= dt / dx * (num_flux[1:] - num_flux[:-1])

    return u

# Solve using different methods for Minmod limiter
u_godunov_minmod = solve(method="godunov", limiter="minmod")
u_roe_minmod = solve(method="roe", limiter="minmod")
u_lax_friedrichs_minmod = solve(method="lax_friedrichs", limiter="minmod")

# Solve using different methods for Superbee limiter
u_godunov_superbee = solve(method="godunov", limiter="superbee")
u_roe_superbee = solve(method="roe", limiter="superbee")
u_lax_friedrichs_superbee = solve(method="lax_friedrichs", limiter="superbee")

# Plot the results for Minmod limiter
plt.figure(figsize=(12, 6))
plt.plot(x, u_initial, 'k--', label="Initial Condition")
plt.plot(x, u_godunov_minmod, label="Godunov (Minmod)", linewidth=4)
plt.plot(x, u_roe_minmod, label="Roe (Minmod)")
plt.plot(x, u_lax_friedrichs_minmod, label="Lax-Friedrichs (Minmod)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Numerical Methods with Minmod Limiter")
plt.legend()
plt.grid()
plt.show()

# Plot the results for Superbee limiter
plt.figure(figsize=(12, 6))
plt.plot(x, u_initial, 'k--', label="Initial Condition")
plt.plot(x, u_godunov_superbee, label="Godunov (Superbee)", linewidth=4)
plt.plot(x, u_roe_superbee, label="Roe (Superbee)")
plt.plot(x, u_lax_friedrichs_superbee, label="Lax-Friedrichs (Superbee)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Numerical Methods with Superbee Limiter")
plt.legend()
plt.grid()
plt.show()
