# -*- coding: utf-8 -*-
"""
seir_simulation.py - SEIR Model with RK4 & simple MPC controller
Author: ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------------
# Model parameters
# --------------------------
mu = 0.000014        # Birth/Death rate per day
beta = 0.5           # Transmission rate
sigma = 0.2          # Incubation rate
gamma = 0.1          # Recovery rate
N = 1_000_000        # Total population

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

        S, E, I, R, B = x_vec
        u1, u2, u3 = u_vec  # u1: transmission reduction, u2: vaccination, u3: treatment

        # SEIR dynamics
        dS_dt = mu * N - beta * (1 - u1) * S * I / N - u2 * S - mu * S
        dE_dt = beta * (1 - u1) * S * I / N - sigma * E - mu * E
        dI_dt = sigma * E - (gamma + u3) * I - mu * I
        dR_dt = (gamma + u3) * I + u2 * S - mu * R
        dB_dt = (gamma + u3) * I  # cumulative removed (or whatever B represents)
        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dB_dt], dtype=float)

# --------------------------
# Measurement class H
# --------------------------
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        return getattr(self, self.measurement_option)(x_vec, u_vec,
                                                      return_measurement_names=return_measurement_names)

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']
        return np.array([x_vec[2], x_vec[3]], dtype=float)

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
        return np.array(x_vec, dtype=float)

# --------------------------
# RK4 integrator
# --------------------------
def rk4_step(f_func, x, u, dt):
    k1 = f_func(x, u)
    k2 = f_func(x + 0.5 * dt * k1, u)
    k3 = f_func(x + 0.5 * dt * k2, u)
    k4 = f_func(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --------------------------
# Simulator with MPC (receding horizon)
# --------------------------
def simulate_seir_rk4_mpc(f_obj, h_obj, tsim_length=365, dt=1.0,
                          x0=None, mpc_horizon_days=14, rterm_u=(1e-3,1e-3,1e-3)):
    if x0 is None:
        x = np.array([N - 1000, 500, 500, 0, 0], dtype=float)
    else:
        x = x0.astype(float).copy()

    t_sim = np.arange(0, tsim_length+dt, dt)
    n_steps = len(t_sim)

    x_log = np.zeros((n_steps, 5))
    u_log = np.zeros((n_steps, 3))
    y_log = []

    def f_func(x_vec, u_vec):
        return f_obj.f(x_vec, u_vec)

    for k in range(n_steps):
        # apply dummy controls (or integrate real MPC later)
        u = np.zeros(3)
        x = rk4_step(f_func, x, u, dt)
        x = np.maximum(x, 0.0)
        y = h_obj.h(x, u)
        x_log[k,:] = x
        u_log[k,:] = u
        y_log.append(y)

    y_log = np.vstack(y_log)
    x_dict = {'S': x_log[:,0], 'E': x_log[:,1], 'I': x_log[:,2], 'R': x_log[:,3], 'B': x_log[:,4]}
    return t_sim, x_dict, u_log, y_log
