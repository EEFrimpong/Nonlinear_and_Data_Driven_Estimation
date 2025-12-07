import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
import pandas as pd
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
m1 = 1.0  # mass 1 (kg)
m2 = 1.0  # mass 2 (kg)
k1 = 10.0  # spring constant 1 (N/m)
k2 = 5.0   # spring constant 2 (N/m)
alpha = 0.5  # nonlinear spring coefficient (N/m^3)

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, m1=m1, m2=m2, k1=k1, k2=k2, alpha=alpha):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for 2-mass nonlinear spring system.
        
        Parameters:
        x_vec : array-like, shape (4,)
            State vector [x1, ẋ1, x2, ẋ2]
        u_vec : array-like, shape (2,)
            Control vector [u1, u2]
        
        Returns:
        x_dot : numpy array, shape (4,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['x1', 'x1_dot', 'x2', 'x2_dot']
        
        # Extract state variables
        x1 = x_vec[0]
        x1_dot = x_vec[1]
        x2 = x_vec[2]
        x2_dot = x_vec[3]

        # Extract control inputs
        u1 = u_vec[0]
        u2 = u_vec[1]
        
        # Compute coupling terms
        displacement = x2 - x1
        linear_coupling = self.k2 * displacement
        nonlinear_coupling = self.alpha * displacement**3
        
        # Dynamics: m₁ẍ₁ = -k₁x₁ + k₂(x₂ - x₁) + α(x₂ - x₁)³ + u₁
        x1_ddot = (-self.k1 * x1 + linear_coupling + nonlinear_coupling + u1) / self.m1
        
        # Dynamics: m₂ẍ₂ = -k₂(x₂ - x₁) - α(x₂ - x₁)³ + u₂
        x2_ddot = (-linear_coupling - nonlinear_coupling + u2) / self.m2
        
        # State derivative vector
        x_dot_vec = np.array([x1_dot, x1_ddot, x2_dot, x2_ddot])
        
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

    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        """Measure both positions"""
        if return_measurement_names:
            return ['x1', 'x2']

        x1 = x_vec[0]
        x2 = x_vec[2]
        
        return np.array([x1, x2])

    def h_velocities(self, x_vec, u_vec, return_measurement_names=False):
        """Measure both velocities"""
        if return_measurement_names:
            return ['x1_dot', 'x2_dot']

        x1_dot = x_vec[1]
        x2_dot = x_vec[3]
        
        return np.array([x1_dot, x2_dot])

    def h_full_state(self, x_vec, u_vec, return_measurement_names=False):
        """Measure all states"""
        if return_measurement_names:
            return ['x1', 'x1_dot', 'x2', 'x2_dot']
        
        return x_vec

    def h_relative_position(self, x_vec, u_vec, return_measurement_names=False):
        """Measure absolute positions and relative displacement"""
        if return_measurement_names:
            return ['x1', 'x2', 'x2_minus_x1']

        x1 = x_vec[0]
        x2 = x_vec[2]
        
        return np.array([x1, x2, x2 - x1])


############################################################################################
# Mass-spring simulation
############################################################################################
def simulate_mass_spring(f, h, tsim_length=20, dt=0.01, measurement_names=None,
                         trajectory_shape='sinusoidal', setpoint=None, rterm=1e-4,
                         measurement_noise_stds=None):
    """
    trajectory_shape: 'sinusoidal', 'step', 'tracking', 'oscillating'
    measurement_noise_stds: dict with keys matching measurement names and values as standard deviations
    """
    # Set state and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2']
    
    # Choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True) 
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names, 
                                   input_names=input_names, measurement_names=measurement_names, 
                                   mpc_horizon=int(1/dt))
    
    # Set measurement noise if provided
    if measurement_noise_stds is not None:
        # Create a measurement noise dictionary that matches the measurement names
        noise_dict = {}
        for meas_name in measurement_names:
            if meas_name in measurement_noise_stds:
                noise_dict[meas_name] = measurement_noise_stds[meas_name]
            else:
                noise_dict[meas_name] = 0.0  # No noise if not specified
        
        # Set the noise in the simulator's model
        for meas_name, std_val in noise_dict.items():
            simulator.model.set_meas_noise(meas_name, std=std_val)

    # Define the set-point(s) to follow
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    if setpoint is None:
        assert trajectory_shape in ['sinusoidal', 'step', 'tracking', 'oscillating']

        if trajectory_shape == 'sinusoidal':
            setpoint = {'x1': 0.5*np.sin(2*np.pi*tsim*0.2),
                        'x1_dot': NA,
                        'x2': 0.3*np.sin(2*np.pi*tsim*0.3 + np.pi/4),
                        'x2_dot': NA,
                       }
        elif trajectory_shape == 'step':
            x1_ref = np.ones_like(tsim) * 0.0
            x1_ref[int(len(tsim)/4):] = 0.5
            x2_ref = np.ones_like(tsim) * 0.0
            x2_ref[int(len(tsim)/2):] = 0.3
            
            setpoint = {'x1': x1_ref,
                        'x1_dot': NA,
                        'x2': x2_ref,
                        'x2_dot': NA,
                       }
        elif trajectory_shape == 'tracking':
            setpoint = {'x1': 0.4*np.cos(2*np.pi*tsim*0.15),
                        'x1_dot': NA,
                        'x2': 0.6*np.cos(2*np.pi*tsim*0.15) + 0.2,
                        'x2_dot': NA,
                       }
        elif trajectory_shape == 'oscillating':
            setpoint = {'x1': 0.3*np.sin(2*np.pi*tsim*0.3),
                        'x1_dot': NA,
                        'x2': 0.5*np.sin(2*np.pi*tsim*0.2) + 0.2*np.cos(2*np.pi*tsim*0.5),
                        'x2_dot': NA,
                       }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function: penalize the squared error between setpoint and actual
    cost_x1 = (simulator.model.x['x1'] - simulator.model.tvp['x1_set']) ** 2
    cost_x2 = (simulator.model.x['x2'] - simulator.model.tvp['x2_set']) ** 2
    cost = cost_x1 + cost_x2

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(u1=rterm, u2=rterm)

    # Set bounds on states and controls (adjust as needed)
    simulator.mpc.bounds['lower', '_x', 'x1'] = -2.0
    simulator.mpc.bounds['upper', '_x', 'x1'] = 2.0
    simulator.mpc.bounds['lower', '_x', 'x2'] = -2.0
    simulator.mpc.bounds['upper', '_x', 'x2'] = 2.0
    simulator.mpc.bounds['lower', '_u', 'u1'] = -50.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 50.0
    simulator.mpc.bounds['lower', '_u', 'u2'] = -50.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 50.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=None, u=None, mpc=True, return_full_output=True)
    
    # Add measurement noise manually if specified
    if measurement_noise_stds is not None:
        for meas_name in measurement_names:
            if meas_name in measurement_noise_stds and meas_name in y_sim:
                noise = np.random.normal(0, measurement_noise_stds[meas_name], len(t_sim))
                y_sim[meas_name] = y_sim[meas_name] + noise

    # Return
    return t_sim, x_sim, u_sim, y_sim, simulator


def package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim):
    """Package simulation data into a pandas DataFrame"""
    # Turn all the sim outputs into pandas dataframes
    df_x = pd.DataFrame(x_sim)  # x_sim is a dict
    df_u = pd.DataFrame(u_sim)  # u_sim is a dict
    df_y = pd.DataFrame(y_sim)  # y_sim is a dict
    df_t = pd.DataFrame({'time': t_sim})  # t_sim is a 1d array, make it a dict
    
    # Rename the columns for y so that they do not conflict with state names
    new_names = {key: 'sensor_' + key for key in df_y}
    df_y = df_y.rename(columns=new_names)
    
    # Merge into a single data frame for the entire trajectory
    df_trajec = pd.concat([df_t, df_x, df_u, df_y], axis=1)
    
    return df_trajec


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Create dynamics and measurement functions
    f_model = F(m1=1.0, m2=1.0, k1=10.0, k2=5.0, alpha=0.5)
    h_model = H(measurement_option='h_full_state')
    
    # Run simulation
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_mass_spring(
        f_model.f, h_model.h, 
        tsim_length=20, 
        dt=0.01, 
        trajectory_shape='sinusoidal',
        rterm=1e-3
    )
    
    # Package data
    df = package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim)
    
    print("Simulation complete!")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
