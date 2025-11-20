# -*- coding: utf-8 -*-
"""
seir_simulation.py - SEIR Model with RK4 & simple MPC controller (no pybounds)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------------
# Model parameters (tweak)
# --------------------------
mu = 0.000014        # Birth/Death rate per day (approximate)
beta = 0.5           # Transmission rate (constant here)
sigma = 0.2          # Incubation rate (1/sigma = incubation period)
gamma = 0.1          # Recovery rate (1/gamma = infectious period)
N = 1_000_000        # Total population (absolute counts)

# --------------------------
# Dynamics class F
# --------------------------
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, mu=mu, beta=beta, sigma=sigma, gamma=gamma, N=N,
          return_state_names=False):
        """
        Continuous time dynamics function for SEIR model.

        x_vec : [S, E, I, R, B]
        u_vec : [u1, u2, u3]
        return_state_names : if True -> return ['S','E','I','R','B']
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'B']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        B = x_vec[4]

        u1 = float(u_vec[0])  # transmission reduction (0..1)
        u2 = float(u_vec[1])  # vaccination rate (0..1 fraction per day)
        u3 = float(u_vec[2])  # increased recovery/treatment rate (0..1 adds to gamma)

        # SEIR dynamics (absolute counts)
        dS_dt = mu * N - beta * (1 - u1) * S * I / N - u2 * S - mu * S
        dE_dt = beta * (1 - u1) * S * I / N - sigma * E - mu * E
        dI_dt = sigma * E - (gamma + u3) * I - mu * I
        dR_dt = (gamma + u3) * I + u2 * S - mu * R
        dB_dt= 0.0
        return np.array([dS_dt, dE_dt, dI_dt, dR_dt,dB_dt], dtype=float)

# --------------------------
# Measurement class H
# --------------------------
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']
        I = x_vec[2]; R = x_vec[3]
        return np.array([I, R], dtype=float)

    def h_i(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute']
        return np.array([x_vec[2]], dtype=float)

    def h_ei(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['E_absolute', 'I_absolute']
        return np.array([x_vec[1], x_vec[2]], dtype=float)

    def h_all(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_absolute', 'E_absolute', 'I_absolute', 'R_absolute', 'B_absolute']
        return x_vec.copy().astype(float)

# --------------------------
# RK4 integrator
# --------------------------
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

# --------------------------
# MPC helper: simulate forward rollout given a candidate control sequence
# --------------------------
def rollout_cost_and_trajectory(x0, u_sequence, f_func, dt, horizon_steps,
                                tvp_I_set=None, tvp_E_set=None,
                                weight_I=100.0, weight_E=10.0,
                                rterm=(1e-3, 1e-3, 1e-3), N=N):
    """
    Simulate forward for horizon_steps using piecewise-constant controls from u_sequence.
    u_sequence is flat array of length 3*horizon_steps: [u1_0,u2_0,u3_0, u1_1,...]
    Returns (cost, traj_states (horizon_steps+1 x 4))
    """
    u_seq = u_sequence.reshape((horizon_steps, 3))
    x = x0.copy()
    traj = [x.copy()]
    total_cost = 0.0

    for k in range(horizon_steps):
        u_k = u_seq[k]
        # propagate one step using RK4
        x = rk4_step(f_func, x, u_k, dt)
        # clip
        x = np.maximum(x, 0.0)
        traj.append(x.copy())
        # compute stage cost: tracking on I and E plus control penalty
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
        total_cost += rterm[0]*u_k[0]**2 + rterm[1]*u_k[1]**2 + rterm[2]*u_k[2]**2

    return total_cost, np.array(traj)

# --------------------------
# MPC optimizer: minimize cost over control sequence
# --------------------------
def solve_mpc(x0, f_func, dt, horizon_steps=14, u_bounds=((0,1),(0,1),(0,1)),
              rterm=(1e-3,1e-3,1e-3), initial_guess=None,
              tvp_I_set=None, tvp_E_set=None):
    """
    Solve a finite-horizon open-loop MPC with simple scipy.minimize.
    - horizon_steps: number of discrete steps in prediction horizon
    - u_bounds: per-control bounds tuple((low,high),...)
    Returns optimal first-step control (u_opt_first) and optionally full sequence.
    """

    n_vars = horizon_steps * 3
    if initial_guess is None:
        # default: zeros
        x0_guess = np.zeros(n_vars)
    else:
        x0_guess = initial_guess.flatten()

    # bounds for minimize
    bounds = []
    for _ in range(horizon_steps):
        for (lo, hi) in u_bounds:
            bounds.append((lo, hi))

    def obj(u_flat):
        cost, _ = rollout_cost_and_trajectory(x0, u_flat, f_func, dt, horizon_steps,
                                             tvp_I_set=tvp_I_set, tvp_E_set=tvp_E_set,
                                             weight_I=100.0, weight_E=10.0, rterm=rterm, N=N)
        return cost

    # simple options
    res = minimize(obj, x0_guess, bounds=bounds, method='SLSQP',
                   options={'maxiter': 200, 'ftol': 1e-3, 'disp': False})

    if not res.success:
        # fallback: return zeros first-step
        u_first = np.zeros(3)
        u_full = np.zeros((horizon_steps, 3))
    else:
        u_full = res.x.reshape((horizon_steps, 3))
        u_first = u_full[0]

    return u_first, u_full

# --------------------------
# High-level simulator that uses RK4 + MPC (receding horizon)
# --------------------------
def simulate_seir_rk4_mpc(f_obj, h_obj, tsim_length=365, dt=1.0,
                          x0=None, mpc_horizon_days=14, mpc_dt=None,
                          rterm_u=(1e-3,1e-3,1e-3)):
    """
    Simulate SEIR with RK4 integrator and a simple MPC controller.
    At each simulation time step, MPC solves for a sequence of controls
    over horizon (mpc_horizon_days) with step dt, then we apply the first control.
    """

    if mpc_dt is None:
        mpc_dt = dt

    horizon_steps = int(np.round(mpc_horizon_days / mpc_dt))
    if horizon_steps < 1:
        horizon_steps = 1

    if x0 is None:
        x = np.array([N - 1000.0, 500.0, 500.0, 0.0, 0.0], dtype=float)
    else:
        x = x0.astype(float).copy()

    # time array
    t_sim = np.arange(0.0, tsim_length + dt/2.0, dt)
    n_steps = len(t_sim)

    # logs
    x_log = np.zeros((n_steps, 5))
    u_log = np.zeros((n_steps, 3))
    y_log = []

    # helper f_func wrapper
    def f_func(x_vec, u_vec):
        return f_obj.f(x_vec, u_vec)

    # define simple target tvp (can be made time-varying)
    # For simplicity use constant target over horizon equal to small fraction of N
    I_set_const = 0.001 * N
    E_set_const = 0.001 * N

    for k in range(n_steps):
        t = t_sim[k]
        # Solve MPC at current state x
        u_first, u_full = solve_mpc(
            x0=x.copy(),
            f_func=f_func,
            dt=dt,
            horizon_steps=horizon_steps,
            u_bounds=((0.0,1.0),(0.0,1.0),(0.0,1.0)),
            rterm=rterm_u,
            initial_guess=None,
            tvp_I_set=np.ones(horizon_steps)*I_set_const,
            tvp_E_set=np.ones(horizon_steps)*E_set_const
        )

        # apply first control for one dt using RK4
        x = rk4_step(f_func, x, u_first, dt)
        x = np.maximum(x, 0.0)

        # measurement
        y = h_obj.h(x, u_first)

        # log
        x_log[k, :] = x
        u_log[k, :] = u_first
        y_log.append(y)

    y_log = np.vstack(y_log) if len(y_log) > 0 else np.zeros((n_steps, 0))

    # return t_sim, x_log dict, u_log, y_log
    x_dict = {'S': x_log[:,0], 'E': x_log[:,1], 'I': x_log[:,2], 'R': x_log[:,3], 'B': x_log[:,4]}
    return t_sim, x_dict, u_log, y_log

# --------------------------
# Example main routine
# --------------------------
def main():
    # initial conditions (absolute)
    x0 = np.array([999500.0, 400.0, 100.0, 0.0, 0.0], dtype=float)

    f_obj = F()
    measurement_options = [
        ('h_ir', 'Primary: I + R'),
        ('h_i', 'I only'),
        ('h_ei', 'E + I'),
        ('h_all', 'S + E + I + R'),
    ]

    results = {}
    for option_name, desc in measurement_options:
        print("\n" + "="*60)
        print(f"Running simulation - measurement: {desc}")
        print("="*60)
        h_obj = H(measurement_option=option_name)
        try:
            t_sim, x_sim, u_sim, y_sim = simulate_seir_rk4_mpc(
                f_obj, h_obj, tsim_length=365, dt=1.0,
                x0=x0, mpc_horizon_days=14, rterm_u=(1e-3,1e-3,1e-3)
            )
            results[option_name] = {'t':t_sim, 'x':x_sim, 'u':u_sim, 'y':y_sim}
            print("Simulation successful.")
            print(f"Final states: S={x_sim['S'][-1]:.0f}, E={x_sim['E'][-1]:.0f}, I={x_sim['I'][-1]:.0f}, R={x_sim['R'][-1]:.0f}")
        except Exception as e:
            print("Simulation failed:", e)

    # Plot the last simulation (h_all) for demonstration
    if 'h_all' in results:
        t = results['h_all']['t']
        x = results['h_all']['x']
        plt.figure(figsize=(12,6))
        plt.plot(t, x['S'], label='S')
        plt.plot(t, x['E'], label='E')
        plt.plot(t, x['I'], label='I')
        plt.plot(t, x['R'], label='R')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('SEIR with RK4 + simple MPC (h_all)')
        plt.legend()
        plt.grid()
        plt.show()

    return results

if __name__ == "__main__":
    main()
