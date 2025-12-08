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
m1 = 1.0      # mass 1 (kg)
m2 = 1.0      # mass 2 (kg)
c1 = 0.5      # damping coefficient 1 (Ns/m)
c2 = 0.5      # damping coefficient 2 (Ns/m)
k1 = 10.0     # spring constant 1 (N/m)
k2 = 5.0      # spring constant 2 (N/m)
alpha1 = 0.5  # nonlinear spring coefficient 1 (N/m^3)
alpha2 = 0.3  # nonlinear spring coefficient 2 (N/m^3)

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, m1=m1, m2=m2, c1=c1, c2=c2, k1=k1, k2=k2, alpha1=alpha1, alpha2=alpha2):
        self.m1 = m1
        self.m2 = m2
        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for 2-mass nonlinear spring-damper system.
        
        State equations:
            m₁q̈₁ = -c₁q̇₁ - k₁q₁ - α₁q₁³ + k₂(q₂ - q₁) + α₂(q₂ - q₁)³
            m₂q̈₂ = -c₂q̇₂ - k₂(q₂ - q₁) - α₂(q₂ - q₁)³
        
        State vector: x = [q₁, q̇₁, q₂, q̇₂] = [x₁, x₂, x₃, x₄]
        
        Parameters:
        x_vec : array-like, shape (4,)
            State vector [q₁, q̇₁, q₂, q̇₂]
        u_vec : array-like, shape (2,)
            Control vector [u1, u2] (optional external forces)
        
        Returns:
        x_dot : numpy array, shape (4,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['q1', 'q1_dot', 'q2', 'q2_dot']
        
        # Extract state variables
        q1 = x_vec[0]       # x₁ = q₁
        q1_dot = x_vec[1]   # x₂ = q̇₁
        q2 = x_vec[2]       # x₃ = q₂
        q2_dot = x_vec[3]   # x₄ = q̇₂

        # Extract control inputs (optional external forces)
        u1 = u_vec[0]
        u2 = u_vec[1]
        
        # Compute coupling terms
        displacement = q2 - q1
        
        # Dynamics for mass 1:
        # ẋ₂ = q̈₁ = (1/m₁)(-c₁q̇₁ - k₁q₁ - α₁q₁³ + k₂(q₂ - q₁) + α₂(q₂ - q₁)³ + u₁)
        q1_ddot = (1.0 / self.m1) * (
            -self.c1 * q1_dot 
            - self.k1 * q1 
            - self.alpha1 * q1**3 
            + self.k2 * displacement 
            + self.alpha2 * displacement**3
            + u1
        )
        
        # Dynamics for mass 2:
        # ẋ₄ = q̈₂ = (1/m₂)(-c₂q̇₂ - k₂(q₂ - q₁) - α₂(q₂ - q₁)³ + u₂)
        q2_ddot = (1.0 / self.m2) * (
            -self.c2 * q2_dot 
            - self.k2 * displacement 
            - self.alpha2 * displacement**3
            + u2
        )
        
        # State derivative vector: ẋ = [q̇₁, q̈₁, q̇₂, q̈₂]
        x_dot_vec = np.array([q1_dot, q1_ddot, q2_dot, q2_ddot])
        
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

    def h_q1(self, x_vec, u_vec, return_measurement_names=False):
        """Measure position q1 only"""
        if return_measurement_names:
            return ['q1']

        q1 = x_vec[0]
        
        return np.array([q1])

    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        """Measure both positions"""
        if return_measurement_names:
            return ['q1', 'q2']

        q1 = x_vec[0]
        q2 = x_vec[2]
        
        return np.array([q1, q2])

    def h_velocities(self, x_vec, u_vec, return_measurement_names=False):
        """Measure both velocities"""
        if return_measurement_names:
            return ['q1_dot', 'q2_dot']

        q1_dot = x_vec[1]
        q2_dot = x_vec[3]
        
        return np.array([q1_dot, q2_dot])

    def h_full_state(self, x_vec, u_vec, return_measurement_names=False):
        """Measure all states"""
        if return_measurement_names:
            return ['q1', 'q1_dot', 'q2', 'q2_dot']
        
        return x_vec

    def h_relative_position(self, x_vec, u_vec, return_measurement_names=False):
        """Measure absolute positions and relative displacement"""
        if return_measurement_names:
            return ['q1', 'q2', 'q2_minus_q1']

        q1 = x_vec[0]
        q2 = x_vec[2]
        
        return np.array([q1, q2, q2 - q1])


############################################################################################
# Mass-spring simulation
############################################################################################
def simulate_mass_spring(f, h, tsim_length=20, dt=0.01, measurement_names=None,
                         trajectory_shape='sinusoidal', setpoint=None, rterm=1e-4,
                         measurement_noise_stds=None, x0=None):
    """
    Simulate 2-mass nonlinear spring-damper system with MPC control.
    
    Parameters:
    -----------
    f : F object
        Dynamics function
    h : H object
        Measurement function
    tsim_length : float
        Simulation length in seconds
    dt : float
        Time step
    measurement_names : list of str
        Names of measurements
    trajectory_shape : str
        'sinusoidal', 'step', 'tracking', 'oscillating'
    setpoint : dict
        Custom setpoint dictionary
    rterm : float
        Control penalty term
    measurement_noise_stds : dict
        Measurement noise standard deviations
    x0 : array-like
        Initial state [q1, q1_dot, q2, q2_dot]
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
    
    # Add measurement noise if provided
    if measurement_noise_stds is not None:
        noise_std_array = []
        for meas in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(meas, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)

    # Define the set-point(s) to follow
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    if setpoint is None:
        assert trajectory_shape in ['sinusoidal', 'step', 'tracking', 'oscillating']

        if trajectory_shape == 'sinusoidal':
            setpoint = {'q1': 0.5*np.sin(2*np.pi*tsim*0.2),
                        'q1_dot': NA,
                        'q2': 0.3*np.sin(2*np.pi*tsim*0.3 + np.pi/4),
                        'q2_dot': NA,
                       }
        elif trajectory_shape == 'step':
            q1_ref = np.ones_like(tsim) * 0.0
            q1_ref[int(len(tsim)/4):] = 0.5
            q2_ref = np.ones_like(tsim) * 0.0
            q2_ref[int(len(tsim)/2):] = 0.3
            
            setpoint = {'q1': q1_ref,
                        'q1_dot': NA,
                        'q2': q2_ref,
                        'q2_dot': NA,
                       }
        elif trajectory_shape == 'tracking':
            setpoint = {'q1': 0.4*np.cos(2*np.pi*tsim*0.15),
                        'q1_dot': NA,
                        'q2': 0.6*np.cos(2*np.pi*tsim*0.15) + 0.2,
                        'q2_dot': NA,
                       }
        elif trajectory_shape == 'oscillating':
            setpoint = {'q1': 0.3*np.sin(2*np.pi*tsim*0.3),
                        'q1_dot': NA,
                        'q2': 0.5*np.sin(2*np.pi*tsim*0.2) + 0.2*np.cos(2*np.pi*tsim*0.5),
                        'q2_dot': NA,
                       }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function: penalize the squared error between setpoint and actual
    cost_q1 = (simulator.model.x['q1'] - simulator.model.tvp['q1_set']) ** 2
    cost_q2 = (simulator.model.x['q2'] - simulator.model.tvp['q2_set']) ** 2
    cost = cost_q1 + cost_q2

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(u1=rterm, u2=rterm)

    # Set bounds on states and controls (adjust as needed)
    simulator.mpc.bounds['lower', '_x', 'q1'] = -2.0
    simulator.mpc.bounds['upper', '_x', 'q1'] = 2.0
    simulator.mpc.bounds['lower', '_x', 'q2'] = -2.0
    simulator.mpc.bounds['upper', '_x', 'q2'] = 2.0
    simulator.mpc.bounds['lower', '_u', 'u1'] = -50.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 50.0
    simulator.mpc.bounds['lower', '_u', 'u2'] = -50.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 50.0

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

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
