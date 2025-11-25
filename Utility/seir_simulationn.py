# -*- coding: utf-8 -*-
"""
seir_simulation_rk4_mpc.py
SEIR Model with 5 states [S,E,I,R,B] using RK4 & simple MPC controller (no pybounds)

Rewritten to follow the same FLOW and API STYLE as the pybounds-based script:
- Global parameters
- F class (dynamics)
- H class (measurements)
- simulate_seir(f, h, ...) with similar signature/arguments
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================================
# Global model parameters (can be tweaked)
# ==========================================================
mu = 0.000014        # Birth/Death rate per day (approximate)
beta = 0.5           # Transmission rate (constant here)
sigma = 0.2          # Incubation rate (1/sigma = incubation period)
gamma = 0.1          # Recovery rate (1/gamma = infectious period)
N = 10_000_000       # Total population (absolute counts)


# ==========================================================
# Dynamics class F  (5-state SEIR + dummy B)
# ==========================================================
class F(object):
    def __init__(self, mu=mu, beta=beta, sigma=sigma, gamma=gamma, N=N):
        """
        SEIR dynamics with 5 states: [S, E, I, R, B]

        NOTE: Here we use a *closed* population with NO births:
              dS/dt = - beta (1-u1) S I / N - u2 S - mu S
        B is a dummy fifth state (can be used later if desired).
        """
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        x_vec: [S,E,I,R,B]
        u_vec: [u1,u2,u3] controls
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'B']

        S, E, I, R, B = x_vec
        u1, u2, u3 = float(u_vec[0]), float(u_vec[1]), float(u_vec[2])

        # Closed population: NO births, only deaths
        lambda_inf = self.beta * (1.0 - u1) * S * I / self.N

        dS_dt = - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - self.sigma * E - self.mu * E
        dI_dt = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R
        dB_dt = 0.0  # placeholder / not used

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dB_dt], dtype=float)


# ==========================================================
# Measurement class H (same pattern as second script)
# ==========================================================
class H(object):
    def __init__(self, measurement_option):
        """
        measurement_option: name of which h_* method to use.
        e.g. 'h_ir', 'h_i', 'h_ei', 'h_all'
        """
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # 1) I and R
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R], dtype=float)

    # 2) I only
    def h_i(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute']
        return np.array([x_vec[2]], dtype=float)

    # 3) E and I
    def h_ei(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['E_absolute', 'I_absolute']
        return np.array([x_vec[1], x_vec[2]], dtype=float)

    # 4) All states
    def h_all(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 'B_absolute']
        return np.array(x_vec, dtype=float)


# ==========================================================
# RK4 integrator
# ==========================================================
def rk4_step(f_func, x, u, dt):
    """
    Single RK4 step: x_{n+1} = x_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    f_func: callable(x_vec, u_vec) -> x_dot
    """
    k1 = f_func(x, u)
    k2 = f_func(x + 0.5 * dt * k1, u)
    k3 = f_func(x + 0.5 * dt * k2, u)
    k4 = f_func(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


# ==========================================================
# MPC helper: rollout + cost
# ==========================================================
def rollout_cost_and_trajectory(x0, u_sequence, f_func, dt, horizon_steps,
                                tvp_I_set=None, tvp_E_set=None,
                                weight_I=100.0, weight_E=10.0,
                                rterm=(1e-3, 1e-3, 1e-3), N=N):
    """
    Simulate forward for horizon_steps using piecewise-constant controls from u_sequence.
    u_sequence: flat array of length 3*horizon_steps: [u1_0,u2_0,u3_0, u1_1,...]

    Returns:
        cost, trajectory (horizon_steps+1 x 5)
    """
    u_seq = u_sequence.reshape((horizon_steps, 3))
    x = x0.copy()
    traj = [x.copy()]
    total_cost = 0.0

    for k in range(horizon_steps):
        u_k = u_seq[k]
        # propagate one step using RK4
        x = rk4_step(f_func, x, u_k, dt)
        x = np.maximum(x, 0.0)
        traj.append(x.copy())

        # cost terms
        I_val = x[2]
        E_val = x[1]

        if tvp_I_set is None:
            I_target = 0.001 * N
        else:
            I_target = float(tvp_I_set[k])

        if tvp_E_set is None:
            E_target = 0.001 * N
        else:
            E_target = float(tvp_E_set[k])

        total_cost += weight_I * (I_val - I_target)**2 + weight_E * (E_val - E_target)**2
        total_cost += rterm[0] * u_k[0]**2 + rterm[1] * u_k[1]**2 + rterm[2] * u_k[2]**2

    return total_cost, np.array(traj)


# ==========================================================
# MPC optimizer
# ==========================================================
def solve_mpc(x0, f_func, dt, horizon_steps=14,
              u_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
              rterm=(1e-3, 1e-3, 1e-3),
              initial_guess=None,
              tvp_I_set=None, tvp_E_set=None):
    """
    Solve a finite-horizon open-loop MPC with scipy.minimize.
    Returns:
        u_first (3,), u_full (horizon_steps x 3)
    """
    n_vars = horizon_steps * 3
    if initial_guess is None:
        x0_guess = np.zeros(n_vars)
    else:
        x0_guess = initial_guess.flatten()

    # bounds
    bounds = []
    for _ in range(horizon_steps):
        for (lo, hi) in u_bounds:
            bounds.append((lo, hi))

    def obj(u_flat):
        cost, _ = rollout_cost_and_trajectory(
            x0, u_flat, f_func, dt, horizon_steps,
            tvp_I_set=tvp_I_set, tvp_E_set=tvp_E_set,
            weight_I=100.0, weight_E=10.0,
            rterm=rterm, N=N
        )
        return cost

    res = minimize(obj, x0_guess, bounds=bounds, method='SLSQP',
                   options={'maxiter': 200, 'ftol': 1e-3, 'disp': False})

    if not res.success:
        u_full = np.zeros((horizon_steps, 3))
        u_first = u_full[0]
    else:
        u_full = res.x.reshape((horizon_steps, 3))
        u_first = u_full[0]

    return u_first, u_full


# ==========================================================
# Simple "simulator" dummy object for API compatibility
# ==========================================================
class SimpleSimulator(object):
    def __init__(self, dt, tsim, state_names, input_names, measurement_names,
                 I_set=None, E_set=None):
        self.dt = dt
        self.tsim = tsim
        self.state_names = state_names
        self.input_names = input_names
        self.measurement_names = measurement_names
        # store targets for reference / plotting
        self.I_set = I_set
        self.E_set = E_set


# ==========================================================
# High-level SEIR simulation with RK4 + MPC
# (API-parallel to the pybounds-based simulate_seir)
# ==========================================================
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):
    """
    SEIR simulation using RK4 + simple MPC (no pybounds), but with an API
    similar to the pybounds-based simulate_seir.

    Inputs:
        f : dynamics function, e.g. F().f
        h : measurement function, e.g. H('h_all').h
        tsim_length : total simulation length (days)
        dt : time step
        measurement_names : optional list of measurement names
        setpoint : dictionary with 'E' and 'I' trajectories (similar idea as second code)
        rterm_u1, rterm_u2, rterm_u3 : control penalties
        x0 : initial state (5-dim)
        measurement_noise_stds : dict mapping measurement name -> std, added to y

    Returns:
        t_sim      : (n_steps,) time array
        x_sim      : dict of state trajectories
        u_sim      : (n_steps, 3) control trajectories
        y_sim      : (n_steps, n_meas) measurement trajectories
        simulator  : SimpleSimulator object (for compatibility)
    """

    # ---------------------------------------
    # Initial conditions
    # ---------------------------------------
    if x0 is None:
        # fractions relative to N, but we store absolute counts
        S0 = 0.70 * N
        E0 = 0.05 * N
        I0 = 0.05 * N
        R0 = N - S0 - E0 - I0
        B0 = 0.0
        x0 = np.array([S0, E0, I0, R0, B0], dtype=float)
    else:
        x0 = np.array(x0, dtype=float)

    # ---------------------------------------
    # State and input names from f
    # ---------------------------------------
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']

    # Measurement names
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # ---------------------------------------
    # Time grid and setpoints (E, I)
    # ---------------------------------------
    t_sim = np.arange(0.0, tsim_length + dt / 2.0, dt)
    n_steps = len(t_sim)

    if setpoint is None:
        # Build simple decaying setpoint on E and I like in the second script
        I_initial = x0[2]
        E_initial = x0[1]

        I_target = 0.0001 * N
        E_target = 0.00005 * N

        I_set = I_target + (I_initial - I_target) * np.exp(-t_sim / 100.0)
        E_set = E_target + (E_initial - E_target) * np.exp(-t_sim / 80.0)
    else:
        # If user provides setpoint dict with arrays 'E' and 'I'
        I_set = np.array(setpoint.get('I', np.zeros_like(t_sim)))
        E_set = np.array(setpoint.get('E', np.zeros_like(t_sim)))

    # ---------------------------------------
    # Logs
    # ---------------------------------------
    x_log = np.zeros((n_steps, 5))
    u_log = np.zeros((n_steps, 3))
    y_log = []

    # MPC horizon (in days) similar to second script
    mpc_horizon_days = 14.0
    horizon_steps = max(1, int(np.round(mpc_horizon_days / dt)))

    # Helper wrapper for f
    def f_func(x_vec, u_vec):
        return f(x_vec, u_vec)

    # ---------------------------------------
    # Main simulation loop
    # ---------------------------------------
    x = x0.copy()

    # noise std array aligned with measurement_names
    if measurement_noise_stds is not None:
        noise_std_array = np.array(
            [measurement_noise_stds.get(mname, 0.0) for mname in measurement_names]
        )
    else:
        noise_std_array = None

    for k in range(n_steps):
        # Build local E/I set for this horizon (slice, clipped at end)
        idx_end = min(n_steps, k + horizon_steps)
        local_len = idx_end - k

        tvp_I_set = I_set[k:idx_end]
        tvp_E_set = E_set[k:idx_end]

        # If near the end and horizon shorter, pad with last value to horizon_steps
        if local_len < horizon_steps:
            tvp_I_set = np.pad(tvp_I_set, (0, horizon_steps - local_len),
                               mode='edge')
            tvp_E_set = np.pad(tvp_E_set, (0, horizon_steps - local_len),
                               mode='edge')

        # Solve MPC at current state
        u_first, u_full = solve_mpc(
            x0=x.copy(),
            f_func=f_func,
            dt=dt,
            horizon_steps=horizon_steps,
            u_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            rterm=(rterm_u1, rterm_u2, rterm_u3),
            initial_guess=None,
            tvp_I_set=tvp_I_set,
            tvp_E_set=tvp_E_set
        )

        # Apply first control
        x = rk4_step(f_func, x, u_first, dt)
        x = np.maximum(x, 0.0)

        # Measurement
        y = h(x, u_first)
        y = np.array(y, dtype=float).ravel()

        # Add noise if specified
        if noise_std_array is not None:
            y = y + np.random.normal(0.0, noise_std_array, size=y.shape)

        # Log
        x_log[k, :] = x
        u_log[k, :] = u_first
        y_log.append(y)

    y_log = np.vstack(y_log) if len(y_log) > 0 else np.zeros((n_steps, 0))

    # Pack states into dict (like pybounds version)
    x_dict = {
        'S': x_log[:, 0],
        'E': x_log[:, 1],
        'I': x_log[:, 2],
        'R': x_log[:, 3],
        'B': x_log[:, 4]
    }

    # Simple simulator object to mimic the pybounds "simulator" return
    simulator = SimpleSimulator(
        dt=dt,
        tsim=t_sim,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        I_set=I_set,
        E_set=E_set
    )

    return t_sim, x_dict, u_log, y_log, simulator


# ==========================================================
# Example main routine (demo)
# ==========================================================
def main():
    # Build F and H objects
    f_obj = F()
    measurement_options = [
        ('h_ir', 'Primary: I + R'),
        ('h_i', 'I only'),
        ('h_ei', 'E + I'),
        ('h_all', 'S + E + I + R + B'),
    ]

    results = {}
    x0 = np.array([9_995_00.0, 400.0, 100.0, 0.0, 0.0], dtype=float)

    for option_name, desc in measurement_options:
        print("\n" + "="*60)
        print(f"Running simulation - measurement: {desc}")
        print("="*60)

        h_obj = H(measurement_option=option_name)

        try:
            t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
                f=f_obj.f,
                h=h_obj.h,
                tsim_length=365,
                dt=1.0,
                measurement_names=None,
                setpoint=None,
                rterm_u1=1e-4,
                rterm_u2=1e-4,
                rterm_u3=1e-4,
                x0=x0,
                measurement_noise_stds=None
            )
            results[option_name] = {'t': t_sim, 'x': x_sim, 'u': u_sim, 'y': y_sim}
            print("Simulation successful.")
            print(f"Final states: S={x_sim['S'][-1]:.0f}, "
                  f"E={x_sim['E'][-1]:.0f}, "
                  f"I={x_sim['I'][-1]:.0f}, "
                  f"R={x_sim['R'][-1]:.0f}, "
                  f"B={x_sim['B'][-1]:.0f}")
        except Exception as e:
            print(f"Simulation failed: {e}")
            import traceback
            traceback.print_exc()

    # Plot the last simulation (h_all) for demonstration
    if 'h_all' in results:
        t = results['h_all']['t']
        x = results['h_all']['x']
        plt.figure(figsize=(12, 6))
        plt.plot(t, x['S'], label='S')
        plt.plot(t, x['E'], label='E')
        plt.plot(t, x['I'], label='I')
        plt.plot(t, x['R'], label='R')
        plt.plot(t, x['B'], label='B')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('5-state SEIR with RK4 + simple MPC (h_all)')
        plt.legend()
        plt.grid()
        plt.show()

    return results


if __name__ == "__main__":
    main()
