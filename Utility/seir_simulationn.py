# -*- coding: utf-8 -*-
"""
SEIR-RK4-MPC.py
5–state SEIR model [S,E,I,R,B] with RK4 and MPC
STRUCTURE MATCHES THE PYBOUNDS SEASONAL CODE
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL PARAMETERS (same style as pybounds code)
# ==========================================================
mu     = 0.02 / 365
beta   = 0.5
sigma  = 0.2
gamma  = 0.1
N      = 1_000_000


# ==========================================================
# MODEL F  — Fixed to match pybounds structure
# ==========================================================
class F(object):
    def __init__(self, mu=mu, beta=beta, sigma=sigma, gamma=gamma, N=N):
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control.
        
        States: x = [S, E, I, R, B]
        Controls: u = [u1, u2, u3]
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'B']

        # Handle None case (for metadata queries)
        if x_vec is None or u_vec is None:
            return None

        S, E, I, R, B = x_vec
        u1, u2, u3 = u_vec

        λ = self.beta * (1 - u1) * S * I / self.N

        dS = -λ - u2*S - self.mu*S
        dE =  λ - self.sigma*E - self.mu*E
        dI =  self.sigma*E - (self.gamma+u3)*I - self.mu*I
        dR =  (self.gamma+u3)*I + u2*S - self.mu*R
        dB =  0.0

        return np.array([dS, dE, dI, dR, dB], float)


# ==========================================================
# MEASUREMENT H — Fixed to match pybounds structure
# ==========================================================
class H(object):
    def __init__(self, option):
        self.option = option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        """Delegate to the specific measurement function"""
        return getattr(self, self.option)(x_vec, u_vec, return_measurement_names)

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: 
            return ['I', 'R']
        if x_vec is None:  # Handle None case
            return None
        return np.array([x_vec[2], x_vec[3]])

    def h_i(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: 
            return ['I']
        if x_vec is None:
            return None
        return np.array([x_vec[2]])

    def h_ei(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: 
            return ['E', 'I']
        if x_vec is None:
            return None
        return np.array([x_vec[1], x_vec[2]])

    def h_all(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: 
            return ['S', 'E', 'I', 'R', 'B']
        if x_vec is None:
            return None
        return x_vec.copy()


# ==========================================================
# RK4
# ==========================================================
def rk4_step(f, x, u, dt):
    """Single RK4 integration step"""
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0


# ==========================================================
# ROLLOUT (same idea as pybounds)
# ==========================================================
def rollout_cost(x0, u_flat, f, dt, Hs, Is,
                 wE=10.0, wI=100.0,
                 r=(1e-3, 1e-3, 1e-3)):
    """
    Compute cost over prediction horizon.
    
    Args:
        x0: Initial state
        u_flat: Flattened control sequence
        f: Dynamics function
        dt: Time step
        Hs: E setpoint trajectory
        Is: I setpoint trajectory
        wE: Weight on E tracking error
        wI: Weight on I tracking error
        r: Control penalty weights (u1, u2, u3)
    """
    Hsteps = len(Hs)
    u_seq = u_flat.reshape(Hsteps, 3)

    x = x0.copy()
    cost = 0.0

    for k in range(Hsteps):
        u = u_seq[k]

        # Penalize control (same style as pybounds)
        cost += r[0]*u[0]**2 + r[1]*u[1]**2 + r[2]*u[2]**2

        # RK4 step
        x = rk4_step(f, x, u, dt)
        x = np.clip(x, 0, N)

        # Penalize E & I tracking errors
        cost += wE*(x[1] - Hs[k])**2 + wI*(x[2] - Is[k])**2

    return cost


# ==========================================================
# MPC SOLVER (same structure as pybounds)
# ==========================================================
def solve_mpc(x, f, dt, horizon, E_set, I_set):
    """
    Solve MPC optimization problem.
    
    Returns:
        u_opt: Optimal first control action
        U_full: Full optimal control sequence
    """
    n = horizon * 3
    u0 = np.zeros(n)

    # Control bounds: [(u1_min, u1_max), (u2_min, u2_max), (u3_min, u3_max)] repeated
    bounds = []
    for _ in range(horizon):
        bounds += [(0, 0.9), (0, 0.6), (0, 0.5)]  # u1, u2, u3 bounds

    # Objective function
    obj = lambda u: rollout_cost(
        x, u, f, dt,
        Hs=E_set[:horizon],
        Is=I_set[:horizon]
    )

    # Solve optimization
    res = minimize(obj, u0, method='SLSQP', bounds=bounds,
                   options={'ftol': 1e-3, 'maxiter': 150, 'disp': False})

    if not res.success:
        print(f"MPC optimization failed: {res.message}")
        U = np.zeros((horizon, 3))
    else:
        U = res.x.reshape(horizon, 3)

    return U[0], U


# ==========================================================
# pybounds-STYLE SIMULATOR OBJECT (dummy for compatibility)
# ==========================================================
class DummySimulator(object):
    """Mimics pybounds.Simulator API for compatibility"""
    def __init__(self):
        self.model = type('', (), {})()
        self.mpc   = type('', (), {})()
        self.bounds = {}
        self.mpc.bounds = self.bounds


# ==========================================================
# MAIN SIMULATION (pybounds-style API)
# ==========================================================
def simulate_seir(f, h, tsim_length=365, dt=1.0,
                  measurement_names=None, setpoint=None,
                  rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):
    """
    Simulate SEIR model with MPC control.
    
    Args:
        f: Dynamics function (from F class)
        h: Measurement function (from H class)
        tsim_length: Simulation duration
        dt: Time step
        measurement_names: List of measurement names
        setpoint: Dict with 'E' and 'I' setpoint trajectories
        rterm_u1, rterm_u2, rterm_u3: Control penalty weights
        x0: Initial state
        measurement_noise_stds: Dict of measurement noise std devs
        
    Returns:
        tsim: Time array
        x_dict: Dictionary of state trajectories
        U: Control input array
        Y: Measurement array
        simulator: DummySimulator object with bounds info
    """
    # Initial state
    if x0 is None:
        S0 = 0.70*N
        E0 = 0.05*N
        I0 = 0.05*N
        R0 = N - S0 - E0 - I0
        B0 = 0.0
        x = np.array([S0, E0, I0, R0, B0], float)
    else:
        x = np.array(x0, float)

    # Get metadata
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # Time grid
    tsim = np.arange(0, tsim_length, dt)
    nT = len(tsim)

    # Build TVP setpoint like pybounds
    if setpoint is None:
        I0_val = x[2]
        E0_val = x[1]
        I_target = 0.0001*N
        E_target = 0.00005*N
        I_set = I_target + (I0_val - I_target)*np.exp(-tsim/100)
        E_set = E_target + (E0_val - E_target)*np.exp(-tsim/80)
    else:
        I_set = setpoint['I']
        E_set = setpoint['E']

    # Allocate storage
    X = np.zeros((nT, 5))
    U = np.zeros((nT, 3))
    Y = np.zeros((nT, len(measurement_names)))

    # MPC horizon
    horizon = int(14/dt)

    # Create dummy simulator matching pybounds API
    simulator = DummySimulator()
    simulator.state_names = state_names
    simulator.input_names = input_names
    simulator.measurement_names = measurement_names

    # Add bounds (same style as pybounds)
    eps = 1e-6
    for s in ['S', 'E', 'I', 'R', 'B']:
        simulator.bounds[('lower', '_x', s)] = eps
        simulator.bounds[('upper', '_x', s)] = N

    simulator.bounds[('lower', '_u', 'u1')] = 0.0
    simulator.bounds[('upper', '_u', 'u1')] = 0.9
    simulator.bounds[('lower', '_u', 'u2')] = 0.0
    simulator.bounds[('upper', '_u', 'u2')] = 0.6
    simulator.bounds[('lower', '_u', 'u3')] = 0.0
    simulator.bounds[('upper', '_u', 'u3')] = 0.5

    # Measurement noise
    if measurement_noise_stds is None:
        noise_std = np.zeros(len(measurement_names))
    else:
        noise_std = np.array([measurement_noise_stds.get(m, 0) 
                              for m in measurement_names])

    # Main simulation loop
    for k in range(nT):
        # Solve MPC
        u_opt, Ufull = solve_mpc(x, f, dt, horizon, E_set[k:], I_set[k:])

        # Apply control
        x = rk4_step(f, x, u_opt, dt)
        x = np.clip(x, 0, N)

        # Measure
        y = h(x, u_opt)
        y = np.array(y, float) + np.random.normal(0, noise_std)

        # Log
        X[k] = x
        U[k] = u_opt
        Y[k] = y

    # Convert to dictionary like pybounds
    x_dict = {
        'S': X[:, 0],
        'E': X[:, 1],
        'I': X[:, 2],
        'R': X[:, 3],
        'B': X[:, 4]
    }

    return tsim, x_dict, U, Y, simulator


# ==========================================================
# DEMO
# ==========================================================
def main():
    """Test simulation with different measurement configurations"""
    
    # Create model instance
    model = F()
    f = model.f

    measurement_options = [
        ('h_ir', 'I + R'),
        ('h_i', 'I only'),
        ('h_ei', 'E + I'),
        ('h_all', 'S,E,I,R,B')
    ]

    print("=" * 60)
    print("SEIR Model Simulation with MPC Control")
    print("=" * 60)

    for key, desc in measurement_options:
        print(f"\nRunning: {desc}")
        h_obj = H(key)
        h = h_obj.h

        t, x, u, y, sim = simulate_seir(
            f, h, 
            tsim_length=100, 
            dt=1.0
        )

        print(f"  Final S: {x['S'][-1]:.1f}")
        print(f"  Final E: {x['E'][-1]:.1f}")
        print(f"  Final I: {x['I'][-1]:.1f}")
        print(f"  Final R: {x['R'][-1]:.1f}")
        print(f"  Total population: {x['S'][-1] + x['E'][-1] + x['I'][-1] + x['R'][-1]:.1f}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    main()
