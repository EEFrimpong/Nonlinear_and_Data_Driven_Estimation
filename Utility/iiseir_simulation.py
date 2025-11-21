# -*- coding: utf-8 -*-
"""
seir_simulation.py - SEIR Model with RK4 & simple MPC controller (no pybounds)
UPDATED: added birth rates, S can go to 0, B state included
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------------
# Model parameters (tweak)
# --------------------------
mu = 0.000014        # Birth/Death rate per day
beta = 0.5           # Transmission rate
sigma = 0.2          # Incubation rate
gamma = 0.1          # Recovery rate
N = 10_000_000       # Total population

# --------------------------
# Dynamics class F
# --------------------------
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, return_state_names=False):
        """SEIR dynamics with 5 states: [S, E, I, R, B]"""
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'B']

        # unpack
        S, E, I, R, B = x_vec
        u1, u2, u3 = float(u_vec[0]), float(u_vec[1]), float(u_vec[2])

        # SEIR equations with births and S→0 control via u2
        dS_dt = mu*N - beta*(1-u1)*S*I/N - u2*S - mu*S
        dE_dt = beta*(1-u1)*S*I/N - sigma*E - mu*E
        dI_dt = sigma*E - (gamma + u3)*I - mu*I
        dR_dt = (gamma + u3)*I + u2*S - mu*R
        dB_dt = 0.0  # B state remains constant

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dB_dt], dtype=float)

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
        I, R = x_vec[2], x_vec[3]
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
    k1 = f_func(x, u)
    k2 = f_func(x + 0.5 * dt * k1, u)
    k3 = f_func(x + 0.5 * dt * k2, u)
    k4 = f_func(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

# --------------------------
# MPC helper and solver remain unchanged (rollout_cost_and_trajectory, solve_mpc)
# ... (use your previous code)
# --------------------------

# --------------------------
# High-level simulator that uses RK4 + MPC
# --------------------------
def simulate_seir_rk4_mpc(f_obj, h_obj, tsim_length=365, dt=1.0,
                          x0=None, mpc_horizon_days=14, mpc_dt=None,
                          rterm_u=(1e-3,1e-3,1e-3)):

    if mpc_dt is None:
        mpc_dt = dt

    horizon_steps = int(np.round(mpc_horizon_days / mpc_dt))
    if horizon_steps < 1:
        horizon_steps = 1

    if x0 is None:
        x = np.array([N - 5000.0, 1000.0, 900.0, 0.0, 0.0], dtype=float)
    else:
        x = x0.astype(float).copy()

    t_sim = np.arange(0.0, tsim_length + dt/2.0, dt)
    n_steps = len(t_sim)

    x_log = np.zeros((n_steps, 5))
    u_log = np.zeros((n_steps, 3))
    y_log = []

    def f_func(x_vec, u_vec):
        return f_obj.f(x_vec, u_vec)

    I_set_const = 0.002 * N
    E_set_const = 0.001 * N

    for k in range(n_steps):
        # Solve MPC at current state
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

        # Apply first control
        x = rk4_step(f_func, x, u_first, dt)
        x = np.maximum(x, 0.0)

        # Measurement
        y = h_obj.h(x, u_first)

        x_log[k, :] = x
        u_log[k, :] = u_first
        y_log.append(y)

    y_log = np.vstack(y_log) if len(y_log) > 0 else np.zeros((n_steps, 0))
    x_dict = {'S': x_log[:,0], 'E': x_log[:,1], 'I': x_log[:,2], 'R': x_log[:,3], 'B': x_log[:,4]}
    return t_sim, x_dict, u_log, y_log

# --------------------------
# Example main routine
# --------------------------
def main():
    x0 = np.array([999500.0, 400.0, 100.0, 0.0, 0.0], dtype=float)
    f_obj = F()
    measurement_options = [
        ('h_ir', 'Primary: I + R'),
        ('h_i', 'I only'),
        ('h_ei', 'E + I'),
        ('h_all', 'S + E + I + R + B'),
    ]

    results = {}
    for option_name, desc in measurement_options:
        print("\n" + "="*60)
        print(f"Running simulation - measurement: {desc}")
        print("="*60)
        h_obj = H(measurement_option=option_name)
        t_sim, x_sim, u_sim, y_sim = simulate_seir_rk4_mpc(
            f_obj, h_obj, tsim_length=365, dt=1.0,
            x0=x0, mpc_horizon_days=14, rterm_u=(1e-3,1e-3,1e-3)
        )
        results[option_name] = {'t':t_sim, 'x':x_sim, 'u':u_sim, 'y':y_sim}
        print("Simulation complete.")
        print(f"Final states: S={x_sim['S'][-1]:.0f}, E={x_sim['E'][-1]:.0f}, "
              f"I={x_sim['I'][-1]:.0f}, R={x_sim['R'][-1]:.0f}, B={x_sim['B'][-1]:.0f}")

    # Plot last simulation (h_all)
    if 'h_all' in results:
        t = results['h_all']['t']
        x = results['h_all']['x']
        plt.figure(figsize=(12,6))
        plt.plot(t, x['S'], label='S')
        plt.plot(t, x['E'], label='E')
        plt.plot(t, x['I'], label='I')
        plt.plot(t, x['R'], label='R')
        plt.plot(t, x['B'], label='B')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title('SEIR with RK4 + MPC (births + B state + S→0)')
        plt.legend()
        plt.grid()
        plt.show()

    return results

if __name__ == "__main__":
    main()
