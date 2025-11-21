# -*- coding: utf-8 -*-
"""
seir_simulation_fixed_subplot.py - SEIR Model with RK4 & fixed controls
All measurement options plotted together
MODIFIED TO DECREASE S COMPARTMENT TO ~2000
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Model parameters - MODIFIED
# --------------------------
mu = 0.000014
beta = 0.8  # INCREASED from 0.5 to 0.8 for higher transmission
sigma = 0.2
gamma = 0.1
N = 10_000_000

# --------------------------
# Dynamics class F
# --------------------------
class F(object):
    def f(self, x_vec, u_vec, return_state_names=False):
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'B']
        S, E, I, R, B = x_vec
        u1, u2, u3 = float(u_vec[0]), float(u_vec[1]), float(u_vec[2])
        dS_dt = mu*N - beta*(1-u1)*S*I/N - u2*S - mu*S
        dE_dt = beta*(1-u1)*S*I/N - sigma*E - mu*E
        dI_dt = sigma*E - (gamma + u3)*I - mu*I
        dR_dt = (gamma + u3)*I + u2*S - mu*R
        dB_dt = 0.0
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
    k2 = f_func(x + 0.5*dt*k1, u)
    k3 = f_func(x + 0.5*dt*k2, u)
    k4 = f_func(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# --------------------------
# Simulation function
# --------------------------
def simulate_seir(f_obj, h_obj, tsim_length=365, dt=1.0, x0=None):
    if x0 is None:
        x = np.array([N - 5000, 1000, 900, 0, 0], dtype=float)
    else:
        x = x0.astype(float).copy()

    t_sim = np.arange(0, tsim_length+dt/2.0, dt)
    n_steps = len(t_sim)
    x_log = np.zeros((n_steps, 5))
    u_log = np.zeros((n_steps, 3))
    y_log = []

    # MODIFIED: Increased u2 to 0.008 for vaccination, set u1=0 for no social distancing
    u_fixed = np.array([0.0, 0.008, 0.0])

    for k in range(n_steps):
        x = rk4_step(f_obj.f, x, u_fixed, dt)
        x = np.maximum(x, 0.0)
        y = h_obj.h(x, u_fixed)
        x_log[k, :] = x
        u_log[k, :] = u_fixed
        y_log.append(y)

    y_log = np.vstack(y_log)
    x_dict = {'S': x_log[:,0], 'E': x_log[:,1], 'I': x_log[:,2], 'R': x_log[:,3], 'B': x_log[:,4]}
    return t_sim, x_dict, u_log, y_log

# --------------------------
# Main routine
# --------------------------
def main():
    # MODIFIED: More initial infected to accelerate epidemic
    x0 = np.array([999500.0, 400.0, 5000.0, 0.0, 0.0], dtype=float)
    f_obj = F()
    measurement_options = [
        ('h_ir', 'Primary: I + R'),
        ('h_i', 'I only'),
        ('h_ei', 'E + I'),
        ('h_all', 'S + E + I + R + B')
    ]

    results = {}
    plt.figure(figsize=(16, 12))
    for i, (option_name, desc) in enumerate(measurement_options, 1):
        h_obj = H(measurement_option=option_name)
        # MODIFIED: Extended simulation time to 800 days
        t_sim, x_sim, u_sim, y_sim = simulate_seir(f_obj, h_obj, tsim_length=800, dt=1.0, x0=x0)
        results[option_name] = {'t': t_sim, 'x': x_sim, 'u': u_sim, 'y': y_sim}

        ax = plt.subplot(2, 2, i)
        meas_names = h_obj.h(x0, np.zeros(3), return_measurement_names=True)
        for j, name in enumerate(meas_names):
            ax.plot(t_sim, y_sim[:,j], label=name)
        ax.set_title(f'Measurements: {desc}')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Population')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot all states
    t = results['h_all']['t']
    x = results['h_all']['x']
    plt.figure(figsize=(12,6))
    for state in ['S','E','I','R','B']:
        plt.plot(t, x[state], label=state)
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    # MODIFIED: Updated title to reflect new control values
    plt.title('SEIR with RK4 + Fixed Controls (u1=0.0, u2=0.008, u3=0)')
    plt.legend()
    plt.grid()
    plt.show()

    # ADDED: Print final S value to verify it reaches ~2000
    print(f"\nFinal S compartment value: {x['S'][-1]:,.0f}")
    print(f"Final R compartment value: {x['R'][-1]:,.0f}")
    print(f"Final I compartment value: {x['I'][-1]:,.0f}")

    return results

# --------------------------
if __name__ == "__main__":
    main()
