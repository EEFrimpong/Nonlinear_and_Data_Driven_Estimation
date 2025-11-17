# -*- coding: utf-8 -*-
"""seir_simulation.py - SEIR Model with MPC Control"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
mu = 0.000014        # Birth/Death rate per day (approximate)
beta = 0.5           # Transmission rate
sigma = 0.2          # Incubation rate (1/sigma = incubation period)
gamma = 0.1          # Recovery rate (1/gamma = infectious period)
N = 1000000          # Total population

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self):
        pass

    def f(self, x_vec, u_vec, mu=mu, beta=beta, sigma=sigma, gamma=gamma, N=N, 
          return_state_names=False):
        """
        Continuous time dynamics function for SEIR model.

        Parameters:
        x_vec : array-like, shape (4,)
            State vector [S, E, I, R]
        u_vec : array-like, shape (3,)
            Control vector [u1, u2, u3]
            u1: transmission reduction (0=no reduction, 1=full reduction)
            u2: vaccination rate
            u3: treatment/isolation rate
        mu : float
            Birth/death rate per day
        beta : float
            Transmission rate
        sigma : float
            Incubation rate (1/sigma = incubation period)
        gamma : float
            Recovery rate (1/gamma = infectious period)
        N : float
            Total population

        Returns:
        x_dot : numpy array, shape (4,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R']

        # Extract state variables
        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]

        # Extract control inputs
        u1 = u_vec[0]  # transmission reduction (social distancing/masks)
        u2 = u_vec[1]  # vaccination rate
        u3 = u_vec[2]  # treatment/isolation rate

        # SEIR dynamics
        dS_dt = mu * N - beta * (1 - u1) * S * I / N - u2 * S - mu * S
        dE_dt = beta * (1 - u1) * S * I / N - sigma * E - mu * E
        dI_dt = sigma * E - (gamma + u3) * I - mu * I
        dR_dt = (gamma + u3) * I + u2 * S - mu * R

        x_dot_vec = np.array([dS_dt, dE_dt, dI_dt, dR_dt])

        return x_dot_vec


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Primary measurement: y = [I, R]^T (Infected and Recovered populations)
        This is the measurement specified in your model
        """
        if return_measurement_names:
            return ['I_absolute', 'R_absolute']

        # Extract state variables
        I = x_vec[2]
        R = x_vec[3]

        # Measurements
        y_vec = np.array([I, R])

        return y_vec

    def h_i(self, x_vec, u_vec, return_measurement_names=False):
        """
        Alternative measurement 1: y = [I] (Infected population only)
        """
        if return_measurement_names:
            return ['I_absolute']

        I = x_vec[2]
        y_vec = np.array([I])

        return y_vec

    def h_ei(self, x_vec, u_vec, return_measurement_names=False):
        """
        Alternative measurement 2: y = [E, I]^T (Exposed and Infected)
        """
        if return_measurement_names:
            return ['E_absolute', 'I_absolute']

        E = x_vec[1]
        I = x_vec[2]
        y_vec = np.array([E, I])

        return y_vec

    def h_all(self, x_vec, u_vec, return_measurement_names=False):
        """
        Alternative measurement 3: y = [S, E, I, R]^T (All compartments)
        """
        if return_measurement_names:
            return ['S_absolute', 'E_absolute', 'I_absolute', 'R_absolute']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        y_vec = np.array([S, E, I, R])

        return y_vec


############################################################################################
# SEIR simulation
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-3, rterm_u2=1e-3, rterm_u3=1e-3, x0=None):
    """
    Simulate SEIR disease model with MPC control

    Parameters:
    -----------
    f : function
        Dynamics function
    h : function
        Measurement function
    tsim_length : float
        Total simulation time in days
    dt : float
        Time step in days
    measurement_names : list
        Names of measurements
    setpoint : dict
        Desired trajectories for states
    rterm_u1 : float
        Control input penalty for transmission reduction
    rterm_u2 : float
        Control input penalty for vaccination
    rterm_u3 : float
        Control input penalty for treatment
    x0 : array-like
        Initial conditions

    Returns:
    --------
    t_sim, x_sim, u_sim, y_sim, simulator
    """
    # Set state and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']  # transmission reduction, vaccination, treatment

    # Choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True)
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names,
                                   input_names=input_names, measurement_names=measurement_names,
                                   mpc_horizon=int(20/dt))

    # Define the time horizon
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    # Define default setpoint if not provided
    if setpoint is None:
        # Get initial conditions
        if x0 is not None:
            I_initial = x0[2]
        else:
            I_initial = 1000

        # Infection setpoint: Decrease infections exponentially
        I_target = 0.001 * N
        I_setpoint = np.maximum(I_initial * np.exp(-tsim / 100), I_target)

        # Exposed setpoint: Decrease exposed exponentially
        E_target = 0.001 * N
        E_setpoint = np.maximum(I_initial * np.exp(-tsim / 100), E_target)

        setpoint = {
            'S': NA,
            'E': E_setpoint,
            'I': I_setpoint,
            'R': NA,
        }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    cost = 100 * cost_I + 10 * cost_E

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'S'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'R'] = N
    
    # Control bounds as specified
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 1.0
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 1.0
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 1.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim, = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
def main():
    """Main function for testing"""
    # Define initial conditions
    # S, E, I, R
    x0 = np.array([
        999500,    # S - Susceptible 
        400,      # E - Exposed
        100,      # I - Infected
        0       # R - Recovered
    ])

    # Create dynamics object
    f_obj = F()

    print("="*80)
    print("TESTING SEIR MODEL WITH DIFFERENT MEASUREMENT OPTIONS")
    print("="*80)

    # Test different measurement options
    measurement_options = [
        ('h_ir', 'Primary: I + R (as specified in your model)'),
        ('h_i', 'Alternative 1: I only'),
        ('h_ei', 'Alternative 2: E + I'),
        ('h_all', 'Alternative 3: S + E + I + R (Full)')
    ]

    results = {}

    for option_name, description in measurement_options:
        print(f"\n{description}")
        print("-" * 60)
        
        h_obj = H(measurement_option=option_name)
        measurement_names = h_obj.h(None, None, return_measurement_names=True)
        print(f"Measurements: {measurement_names}")
        
        try:
            t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
                f_obj.f, h_obj.h, tsim_length=365, dt=1.0, x0=x0
            )
            results[option_name] = {
                't': t_sim,
                'x': x_sim,
                'u': u_sim,
                'y': y_sim,
                'simulator': simulator,
                'measurements': measurement_names
            }
            print(f"✓ Simulation successful")
            print(f"  Final S: {x_sim['S'][-1]:.0f}")
            print(f"  Final E: {x_sim['E'][-1]:.0f}")
            print(f"  Final I: {x_sim['I'][-1]:.0f}")
            print(f"  Final R: {x_sim['R'][-1]:.0f}")
        except Exception as e:
            print(f"✗ Simulation failed: {str(e)}")

    print("\n" + "="*80)
    print("SEIR MODEL SIMULATION COMPLETE")
    print("="*80)
    print("\nModel parameters:")
    print(f"  μ (birth/death rate): {mu}")
    print(f"  β (transmission rate): {beta}")
    print(f"  σ (incubation rate): {sigma} (incubation period: {1/sigma:.1f} days)")
    print(f"  γ (recovery rate): {gamma} (infectious period: {1/gamma:.1f} days)")
    print(f"  N (population): {N}")
    print("\nControl inputs:")
    print("  u1: Transmission reduction (social distancing/masks), 0 ≤ u1 ≤ 1")
    print("  u2: Vaccination rate, 0 ≤ u2 ≤ 1")
    print("  u3: Treatment/isolation rate, 0 ≤ u3 ≤ 1")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
