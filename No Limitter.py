import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
x_min, x_max = -5.0, 5.0  # Domain
N = 100  # Number of grid points
dx = (x_max - x_min) / N  # Grid spacing
x = np.linspace(x_min + dx / 2, x_max - dx / 2, N)  # Cell centers
dt = 0.01  # Time step
T = 2.0  # Final time
num_steps = int(T / dt)

# Initial condition
u_initial = np.zeros(N, dtype=np.float64)
u_initial[(x >= -1) & (x < 0)] = -1.0
u_initial[(x >= 0) & (x < 1)] = 1.0

# Flux function
def flux(u):
    return 0.5 * u**2

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

# Solver function without slope limiter
def solve_no_limiter(method="godunov"):
    u = u_initial.copy()
    for n in range(num_steps):
        # Extend u to ghost cells
        u_ext = np.zeros(N + 2, dtype=np.float64)
        u_ext[1:-1] = u
        u_ext[0] = u[0]  # Left ghost cell (copy boundary)
        u_ext[-1] = u[-1]  # Right ghost cell (copy boundary)

        # Compute left and right states at cell interfaces (no slope limiter)
        u_left = u_ext[:-1]
        u_right = u_ext[1:]

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

# Solve using different methods without slope limiter
u_godunov_no_limiter = solve_no_limiter(method="godunov")
u_roe_no_limiter = solve_no_limiter(method="roe")
u_lax_friedrichs_no_limiter = solve_no_limiter(method="lax_friedrichs")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, u_initial, 'k--', label="Initial Condition")
plt.plot(x, u_godunov_no_limiter, label="Godunov (No Limiter)", linewidth=4)
plt.plot(x, u_roe_no_limiter, label="Roe (No Limiter)")
plt.plot(x, u_lax_friedrichs_no_limiter, label="Lax-Friedrichs (No Limiter)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Numerical Methods Without Slope Limiter")
plt.legend()
plt.grid()
plt.show()