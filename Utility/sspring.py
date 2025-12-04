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
k1 = 2.0  # spring constant 1 (N/m)
k2 = 3.0  # spring constant 2 (N/m)

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, m1=m1, m2=m2, k1=k1, k2=k2):
        """
        Initialize with system parameters.
        """
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for two-mass spring system.
        
        Control-affine form: ẋ = f₀(x) + f₁(x)u₁ + f₂(x)u₂
        
        States:
            x = [x1, x2, x3, x4]
            x1: position of mass 1
            x2: velocity of mass 1
            x3: position of mass 2
            x4: velocity of mass 2
        
        Controls:
            u = [u1, u2]
            u1: force on mass 1
            u2: force on mass 2
        
        Dynamics:
            ẋ₁ = x₂
            ẋ₂ = -(k₁+k₂)/m₁·x₁ + k₂/m₁·x₃ + u₁/m₁
            ẋ₃ = x₄
            ẋ₄ = k₂/m₂·x₁ - k₂/m₂·x₃ + u₂/m₂
        
        Returns:
            x_dot : numpy array, shape (4,)
                Time derivative of state vector
        """
        if return_state_names:
            return ['x1', 'x2', 'x3', 'x4']
        
        # Extract state variables
        x1 = x_vec[0]
        x2 = x_vec[1]
        x3 = x_vec[2]
        x4 = x_vec[3]
        
        # Extract control inputs
        u1 = u_vec[0]
        u2 = u_vec[1]
        
        # f₀ component: drift dynamics (no controls)
        f0_contribution = np.array([
            x2,
            -(self.k1 + self.k2)/self.m1 * x1 + (self.k2/self.m1) * x3,
            x4,
            (self.k2/self.m2) * x1 - (self.k2/self.m2) * x3
        ])
        
        # f₁ component: multiplied by control u1
        f1_contribution = u1 * np.array([
            0,
            1.0/self.m1,
            0,
            0
        ])
        
        # f₂ component: multiplied by control u2
        f2_contribution = u2 * np.array([
            0,
            0,
            0,
            1.0/self.m2
        ])
        
        # Combined dynamics
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution
        
        return x_dot_vec


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option):
        """
        Initialize measurement function.
        
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # -------------------------------------------------------------------------
    # 1. h_x1_only: Measure only position of mass 1
    # -------------------------------------------------------------------------
    def h_x1_only(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [x1]
        This is the standard observability case.
        """
        if return_measurement_names:
            return ['x1_measured']
        
        x1 = x_vec[0]
        return np.array([x1])

    # -------------------------------------------------------------------------
    # 2. h_positions: Measure both positions
    # -------------------------------------------------------------------------
    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [x1, x3]^T
        """
        if return_measurement_names:
            return ['x1_measured', 'x3_measured']
        
        x1 = x_vec[0]
        x3 = x_vec[2]
        return np.array([x1, x3])

    # -------------------------------------------------------------------------
    # 3. h_mass1: Measure position and velocity of mass 1
    # -------------------------------------------------------------------------
    def h_mass1(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [x1, x2]^T
        """
        if return_measurement_names:
            return ['x1_measured', 'x2_measured']
        
        x1 = x_vec[0]
        x2 = x_vec[1]
        return np.array([x1, x2])

    # -------------------------------------------------------------------------
    # 4. h_full_state: Measure all states
    # -------------------------------------------------------------------------
    def h_full_state(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [x1, x2, x3, x4]^T
        """
        if return_measurement_names:
            return ['x1_measured', 'x2_measured', 'x3_measured', 'x4_measured']
        
        return np.array(x_vec)

    # -------------------------------------------------------------------------
    # 5. h_with_forces: Measure positions and include force measurements
    # -------------------------------------------------------------------------
    def h_with_forces(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [x1, x3, f1, f2]^T
        where f1, f2 are the spring forces
        """
        if return_measurement_names:
            return ['x1_measured', 'x3_measured', 'force1', 'force2']
        
        x1 = x_vec[0]
        x3 = x_vec[2]
        
        # Spring forces (from dynamics)
        force1 = -(k1 + k2) * x1 + k2 * x3
        force2 = k2 * x1 - k2 * x3
        
        return np.array([x1, x3, force1, force2])


############################################################################################
# Spring system simulation with MPC
############################################################################################
def simulate_spring(f, h, tsim_length=10, dt=0.1, measurement_names=None,
                    setpoint=None, rterm_u1=0.01, rterm_u2=0.01,
                    x0=None, measurement_noise_stds=None):
    """
    Simulate two-mass spring system with MPC control.
    
    Parameters:
    -----------
    f : F object
        Dynamics function
    h : H object
        Measurement function
    tsim_length : float
        Simulation time in seconds
    dt : float
        Time step
    measurement_names : list
        Names of measurements
    setpoint : dict
        Desired trajectories for states
    rterm_u1, rterm_u2 : float
        Control penalty weights
    x0 : array
        Initial state [x1, x2, x3, x4]
    measurement_noise_stds : dict
        Standard deviations for measurement noise
    
    Returns:
    --------
    t_sim : array
        Time points
    x_sim : dict
        State trajectories
    u_sim : dict
        Control inputs
    y_sim : dict
        Measurements
    simulator : pybounds.Simulator
        Simulator object
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
    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(2.0/dt)  # 2 second horizon
    )
    
    # Add measurement noise if provided
    if measurement_noise_stds is not None:
        noise_std_array = []
        for meas in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(meas, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)
    
    # Time grid
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)
    
    # Default initial conditions
    if x0 is None:
        x0 = np.array([1.0, 0.0, 0.5, 0.0])
    
    # Default setpoint: exponential decay to rest at origin
    if setpoint is None:
        x1_target = 0.0
        x3_target = 0.0
        
        x1_set = x1_target + (x0[0] - x1_target) * np.exp(-tsim / 3.0)
        x3_set = x3_target + (x0[2] - x3_target) * np.exp(-tsim / 3.0)
        
        setpoint = {
            'x1': x1_set,
            'x2': NA,
            'x3': x3_set,
            'x4': NA
        }
    
    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')
    
    # Define MPC cost function: penalize deviation from setpoint
    cost_x1 = (simulator.model.x['x1'] - simulator.model.tvp['x1_set']) ** 2
    cost_x2 = (simulator.model.x['x2'] - simulator.model.tvp['x2_set']) ** 2
    cost_x3 = (simulator.model.x['x3'] - simulator.model.tvp['x3_set']) ** 2
    cost_x4 = (simulator.model.x['x4'] - simulator.model.tvp['x4_set']) ** 2
    
    cost = 100.0 * cost_x1 + 10.0 * cost_x2 + 100.0 * cost_x3 + 10.0 * cost_x4
    
    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    
    # Set input penalty
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2)
    
    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'x1'] = -5.0
    simulator.mpc.bounds['upper', '_x', 'x1'] = 5.0
    simulator.mpc.bounds['lower', '_x', 'x2'] = -10.0
    simulator.mpc.bounds['upper', '_x', 'x2'] = 10.0
    simulator.mpc.bounds['lower', '_x', 'x3'] = -5.0
    simulator.mpc.bounds['upper', '_x', 'x3'] = 5.0
    simulator.mpc.bounds['lower', '_x', 'x4'] = -10.0
    simulator.mpc.bounds['upper', '_x', 'x4'] = 10.0
    
    # Control bounds
    simulator.mpc.bounds['lower', '_u', 'u1'] = -5.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 5.0
    simulator.mpc.bounds['lower', '_u', 'u2'] = -5.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 5.0
    
    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )
    
    # Return
    return t_sim, x_sim, u_sim, y_sim, simulator


def package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim):
    """
    Turn all the simulation outputs into pandas dataframes.
    """
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
