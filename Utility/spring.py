# ==============================================================
# TWO-MASS SPRING SYSTEM WITH MPC AND KALMAN FILTER
# Following the pybounds/SEIR architecture
# ==============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import pybounds

# ==============================================================
# GLOBAL PARAMETERS
# ==============================================================
m1 = 1.0
m2 = 1.0
k1 = 2.0
k2 = 3.0

# ==============================================================
# CONTINUOUS TIME DYNAMICS CLASS
# ==============================================================
class F(object):
    """
    Dynamics for two-mass spring system with control.
    
    States: x = [x1, x2, x3, x4]
        x1: position of mass 1
        x2: velocity of mass 1
        x3: position of mass 2
        x4: velocity of mass 2
    
    Controls: u = [u1, u2]
        u1: force on mass 1
        u2: force on mass 2
    """
    
    def __init__(self, m1=m1, m2=m2, k1=k1, k2=k2):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
    
    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function.
        
        dx1/dt = x2
        dx2/dt = -(k1+k2)/m1 * x1 + k2/m1 * x3 + u1/m1
        dx3/dt = x4
        dx4/dt = k2/m2 * x1 - k2/m2 * x3 + u2/m2
        """
        if return_state_names:
            return ['x1', 'x2', 'x3', 'x4']
        
        x1, x2, x3, x4 = x_vec
        u1, u2 = u_vec
        
        dx1 = x2
        dx2 = -(self.k1 + self.k2)/self.m1 * x1 + (self.k2/self.m1) * x3 + u1/self.m1
        dx3 = x4
        dx4 = (self.k2/self.m2) * x1 - (self.k2/self.m2) * x3 + u2/self.m2
        
        return np.array([dx1, dx2, dx3, dx4])


# ==============================================================
# MEASUREMENT CLASS
# ==============================================================
class H(object):
    """
    Measurement functions for the two-mass system.
    """
    
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option
    
    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)
    
    # Option 1: Measure only position of mass 1
    def h_x1_only(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1_measured']
        return np.array([x_vec[0]])
    
    # Option 2: Measure both positions
    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1_measured', 'x3_measured']
        return np.array([x_vec[0], x_vec[2]])
    
    # Option 3: Measure all states (full state feedback)
    def h_full_state(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1_measured', 'x2_measured', 'x3_measured', 'x4_measured']
        return np.array(x_vec)
    
    # Option 4: Measure position and velocity of mass 1
    def h_mass1(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1_measured', 'x2_measured']
        return np.array([x_vec[0], x_vec[1]])


# ==============================================================
# SIMULATION FUNCTION
# ==============================================================
def simulate_spring(f, h, tsim_length=20, dt=0.02, measurement_names=None,
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
        Initial state
    measurement_noise_stds : dict
        Standard deviations for measurement noise
    """
    
    # Default initial conditions
    if x0 is None:
        x0 = np.array([1.0, 0.0, 0.5, 0.0])
    
    # State and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2']
    
    # Measurement names
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)
    
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
    
    # Default setpoint: bring both masses to rest at origin
    if setpoint is None:
        # Exponential decay to zero for positions
        x1_target = 0.0
        x3_target = 0.0
        
        x1_set = x1_target + (x0[0] - x1_target) * np.exp(-tsim / 3.0)
        x3_set = x3_target + (x0[2] - x3_target) * np.exp(-tsim / 3.0)
        
        setpoint = {
            'x1': x1_set,
            'x2': np.zeros_like(tsim),
            'x3': x3_set,
            'x4': np.zeros_like(tsim)
        }
    
    simulator.update_dict(setpoint, name='setpoint')
    
    # MPC cost function: penalize deviation from setpoint
    cost = (
        100.0 * (simulator.model.x['x1'] - simulator.model.tvp['x1_set'])**2 +
        10.0 * (simulator.model.x['x2'] - simulator.model.tvp['x2_set'])**2 +
        100.0 * (simulator.model.x['x3'] - simulator.model.tvp['x3_set'])**2 +
        10.0 * (simulator.model.x['x4'] - simulator.model.tvp['x4_set'])**2
    )
    
    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    
    # Input penalties
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2)
    
    # State bounds
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
    
    # Run simulation with MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )
    
    return t_sim, x_sim, u_sim, y_sim, simulator


# ==============================================================
# MAIN SIMULATION
# ==============================================================
if __name__ == "__main__":
    
    print("="*60)
    print("TWO-MASS SPRING SYSTEM WITH MPC")
    print("="*60)
    
    # Create dynamics and measurement objects
    f = F(m1=m1, m2=m2, k1=k1, k2=k2)
    h = H(measurement_option='h_x1_only')  # Only measure x1
    
    # Initial conditions
    x0 = np.array([1.0, 0.0, 0.5, 0.0])
    
    # Measurement noise
    measurement_noise_stds = {
        'x1_measured': 0.05
    }
    
    # Run simulation
    print("\nRunning MPC simulation...")
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_spring(
        f, h,
        tsim_length=20.0,
        dt=0.02,
        x0=x0,
        measurement_noise_stds=measurement_noise_stds,
        rterm_u1=0.01,
        rterm_u2=0.01
    )
    
    print(f"Simulation completed: {len(t_sim)} time steps")
    print(f"Final state: x1={x_sim[0,-1]:.4f}, x2={x_sim[1,-1]:.4f}, "
          f"x3={x_sim[2,-1]:.4f}, x4={x_sim[3,-1]:.4f}")
    
    # ==============================================================
    # PLOTTING
    # ==============================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Positions
    axes[0].plot(t_sim, x_sim[0,:], 'b-', linewidth=2, label='x₁ (mass 1)')
    axes[0].plot(t_sim, x_sim[2,:], 'r-', linewidth=2, label='x₃ (mass 2)')
    if y_sim is not None:
        axes[0].plot(t_sim, y_sim[0,:], 'k.', alpha=0.2, markersize=2, 
                    label='Measured x₁')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    axes[0].set_ylabel('Position', fontsize=11)
    axes[0].set_title('Two-Mass Spring System with MPC Control', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Velocities
    axes[1].plot(t_sim, x_sim[1,:], 'b-', linewidth=2, label='x₂ (velocity 1)')
    axes[1].plot(t_sim, x_sim[3,:], 'r-', linewidth=2, label='x₄ (velocity 2)')
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    axes[1].set_ylabel('Velocity', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Control inputs
    axes[2].plot(t_sim[:-1], u_sim[0,:], 'g-', linewidth=2, label='u₁ (force on mass 1)')
    axes[2].plot(t_sim[:-1], u_sim[1,:], 'm-', linewidth=2, label='u₂ (force on mass 2)')
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Control Force', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Phase portrait
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mass 1 phase portrait
    axes2[0].plot(x_sim[0,:], x_sim[1,:], 'b-', linewidth=2)
    axes2[0].plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    axes2[0].plot(x_sim[0,-1], x_sim[1,-1], 'ro', markersize=10, label='End')
    axes2[0].set_xlabel('Position x₁', fontsize=11)
    axes2[0].set_ylabel('Velocity x₂', fontsize=11)
    axes2[0].set_title('Phase Portrait - Mass 1', fontsize=12, fontweight='bold')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Mass 2 phase portrait
    axes2[1].plot(x_sim[2,:], x_sim[3,:], 'r-', linewidth=2)
    axes2[1].plot(x0[2], x0[3], 'go', markersize=10, label='Start')
    axes2[1].plot(x_sim[2,-1], x_sim[3,-1], 'ro', markersize=10, label='End')
    axes2[1].set_xlabel('Position x₃', fontsize=11)
    axes2[1].set_ylabel('Velocity x₄', fontsize=11)
    axes2[1].set_title('Phase Portrait - Mass 2', fontsize=12, fontweight='bold')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
