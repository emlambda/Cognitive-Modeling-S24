import numpy as np
import matplotlib.pyplot as plt
def simulate_diffusion(v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    v     : float
        The drift rate (rate of information uptake)
    a     : float
        The boundary separation (decision threshold).
    beta  : float in [0, 1]
        Relative starting point (prior option preferences)
    tau   : float
        Non-decision time (additive constant)
    dt    : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time: float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a 
        binary decision (c - int) 
    """

    # Inits (process starts at relative starting point)
    y = beta * a
    num_steps = tau
    const = scale*np.sqrt(dt)

    # Loop through process and check boundary conditions
    while (y <= a and y >= 0) and num_steps <= max_time:

        # Perform diffusion equation
        z = np.random.randn()
        y += v*dt + const*z

        # Increment step counter
        num_steps += dt

    if y >= a:
        c = 1
    else:
        c = 0
    return np.array((round(num_steps, 3), c))

def simulate_diffusion_n(num_sims, v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """Add a nice docstring."""

    data = np.zeros((num_sims, 2))
    for n in range(num_sims):
        data[n, :] = simulate_diffusion(v, a, beta, tau, dt, scale, max_time)
    return np.array(data)

data = []
upper_mean = []
lower_mean = []
for a in np.linspace(1,4,25):
    params = {
        'v': 0,
        'a': a,
        'beta': 0.5,
        'tau': 0.1
    }
    current = simulate_diffusion_n(2000,**params)
    while (sum(current[:,1]==1) == 0) or (sum(current[:,1]==0) == 0): #keep generating data if upper or lower bound not reached
        print("Didn't reach upper or lower")
        current = simulate_diffusion_n(2000,**params)
    print(sum(current[:,1]==0))
    upper_mean.append(np.mean(current[current[:,1]==1,0]))
    lower_mean.append(np.mean(current[current[:,1]==0,0]))

difference = np.abs(np.array(upper_mean)-np.array(lower_mean))
# Plot mean differences
plt.plot(np.linspace(1,4,25), difference, marker='o')
plt.xlabel('Boundary difference')
plt.ylabel('Mean Difference (Upper - Lower)')
plt.ylim(0,max(difference))
plt.title('Mean Difference vs Boundary Difference')
plt.grid(True)
plt.show()
