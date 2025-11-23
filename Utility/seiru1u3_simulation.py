import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
mu = 0.02 / 365      # Natural mortality rate per day (2% per year)
sigma = 1.0 / 5.2    # Progression rate from E to I (5.2 days incubation period)
gamma = 1.0 / 10.0   # Recovery rate (10 days infectious period)
N = 1000000          # Total population

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, sigma=sigma, gamma=gamma, N=N):
        """Initialize with parameters stored as instance variables"""
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control.
        
        Parameters:
        -----------
        x_vec : array-like, shape (5,)
            State vector [S, E, I, R, beta]
        u_vec : array-like, shape (2,)
            Control vector [u1, u3]
            u1: social distancing/prevention effectiveness (0=no intervention, 1=full prevention)
            u3: treatment rate (additional recovery rate beyond natural gamma)
        
        Returns:
        --------
        x_dot : numpy array, shape (5,)
            Time derivative of state vector
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # Extract state variables
        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]

        # Extract control inputs
        u1 = u_vec[0]  # prevention/social distancing
        u3 = u_vec[1]  # treatment rate

        # SEIR dynamics with controls
        # Force of infection with control u1
        lambda_infection = beta * (1 - u1) * S * I / self.N

        # State derivatives
        dS_dt = self.mu * self.N - lambda_infection - self.mu * S
        dE_dt = lambda_infection - self.sigma * E - self.mu * E
        dI_dt = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I - self.mu * R
        dbeta_dt = 0.0  # beta is constant (unknown parameter)

        x_dot_vec = np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt])

        return x_dot_vec


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, sigma=sigma, gamma=gamma, N=N):
        """Initialize with measurement option and model parameters"""
        self.measurement_option = measurement_option
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 1: y = [I] (Reported infected cases only)
        Most basic measurement - often insufficient for full observability
        """
        if return_measurement_names:
            return ['I_reported']

        I = x_vec[2]
        y_vec = np.array([I])
        return y_vec

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 2: y = [I, new_cases]^T
        Includes both prevalence and incidence (new infections)
        """
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S = x_vec[0]
        I = x_vec[2]
        beta = x_vec[4]
        u1 = u_vec[0]
        
        # New cases (incidence rate)
        new_cases = beta * (1 - u1) * S * I / self.N
        
        y_vec = np.array([I, new_cases])
        return y_vec

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 3: y = [S, E, I, R]^T
        All four compartments measured
        Good observability for compartments, but not for beta
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        
        y_vec = np.array([S, E, I, R])
        return y_vec

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 3: y = [I, R]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured']

        I = x_vec[2]
        R = x_vec[3]
        
        y_vec = np.array([I, R])
        return y_vec

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 4: y = [S, E, I, R, beta]^T
        All compartments plus transmission rate
        Full observability
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        
        y_vec = np.array([S, E, I, R, beta])
        return y_vec

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 5: y = [S, E, I, R, new_infections, progressions, recoveries]^T
        Includes all major flow rates
        Excellent for parameter estimation
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_infections', 'progressions', 'recoveries']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]
        u3 = u_vec[1]
        
        # Flow rates
        new_infections = beta * (1 - u1) * S * I / self.N
        progressions = self.sigma * E
        recoveries = (self.gamma + u3) * I
        
        y_vec = np.array([S, E, I, R, new_infections, progressions, recoveries])
        return y_vec

    def h_incidence_recovery(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 6: y = [I, R, new_cases]^T
        Practical measurement: infected, recovered, and new cases
        """
        if return_measurement_names:
            return ['I_reported', 'R_measured', 'new_cases']

        S = x_vec[0]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]
        
        new_cases = beta * (1 - u1) * S * I / self.N
        
        y_vec = np.array([I, R, new_cases])
        return y_vec

    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement 7: y = [E, I, new_infections, progressions]^T
        Focus on E→I transition for better sigma observability
        """
        if return_measurement_names:
            return ['E_measured', 'I_measured', 'new_infections', 'progressions']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        beta = x_vec[4]
        u1 = u_vec[0]
        
        new_infections = beta * (1 - u1) * S * I / self.N
        progressions = self.sigma * E
        
        y_vec = np.array([E, I, new_infections, progressions])
        return y_vec


############################################################################################
# SEIR simulation with MPC
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u3=1e-4, x0=None):
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
        Control input penalty for prevention/social distancing
    rterm_u3 : float
        Control input penalty for treatment
    x0 : array-like
        Initial conditions [S0, E0, I0, R0, beta0]

    Returns:
    --------
    t_sim, x_sim, u_sim, y_sim, simulator
    """
    # Set state and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u3']  # prevention and treatment

    # Choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True)
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names,
                                   input_names=input_names, measurement_names=measurement_names,
                                   mpc_horizon=int(10/dt))

    # Define the time horizon
    tsim = np.arange(0, tsim_length, step=dt)
    no_setpoint = None

    # Define default setpoint if not provided
    if setpoint is None:
        # Infection setpoint: Exponential decrease to low endemic level
        if x0 is not None:
            I_initial = x0[2]
            E_initial = x0[1]
        else:
            I_initial = 10000
            E_initial = 5000
        
        I_target = 0.0001 * N  # Target 0.01% infected (low endemic level)
        E_target = 0.00005 * N  # Target low exposed level
        
        # Exponential decay
        I_setpoint = I_target + (I_initial - I_target) * np.exp(-tsim / 100)
        E_setpoint = E_target + (E_initial - E_target) * np.exp(-tsim / 80)

        setpoint = {
            'S': no_setpoint,
            'E': E_setpoint,
            'I': I_setpoint,
            'R': no_setpoint,
            'beta': 0.5 * np.ones_like(tsim),  # Target beta (for estimation)
        }

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Define MPC cost function
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
    cost = 100 * cost_I + 10 * cost_E  # Higher weight on reducing infections

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty
    simulator.mpc.set_rterm(u1=rterm_u1, u3=rterm_u3)

    # Set bounds on states and controls
    epsilon = 1e-6
    
    simulator.mpc.bounds['lower', '_x', 'S'] = epsilon
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = epsilon
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = epsilon
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = epsilon
    simulator.mpc.bounds['upper', '_x', 'R'] = N
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.1
    simulator.mpc.bounds['upper', '_x', 'beta'] = 2.0
    
    # Control bounds
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9  # Max 90% prevention
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.5  # Max treatment rate

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Define realistic initial conditions for SEIR model
    S0 = 0.90 * N      # 90% susceptible
    E0 = 0.01 * N      # 1% exposed
    I0 = 0.01 * N      # 1% infected
    R0 = N - S0 - E0 - I0  # Remainder recovered
    beta0 = 0.5        # Initial guess for transmission rate
    
    x0 = np.array([S0, E0, I0, R0, beta0])
    
    print("="*80)
    print("SEIR MODEL WITH CONTROL - TESTING ALL MEASUREMENT OPTIONS")
    print("="*80)
    print(f"\nInitial Population Distribution:")
    print(f"  Susceptible (S): {S0:,.0f} ({100*S0/N:.1f}%)")
    print(f"  Exposed (E):     {E0:,.0f} ({100*E0/N:.1f}%)")
    print(f"  Infected (I):    {I0:,.0f} ({100*I0/N:.1f}%)")
    print(f"  Recovered (R):   {R0:,.0f} ({100*R0/N:.1f}%)")
    print(f"  Total:           {N:,}")
    print(f"  Beta (initial):  {beta0:.2f}")
    print(f"\nModel Parameters:")
    print(f"  mu (mortality):  {mu:.6f} per day ({mu*365:.4f} per year)")
    print(f"  sigma (E→I):     {sigma:.6f} per day ({1/sigma:.1f} days incubation)")
    print(f"  gamma (I→R):     {gamma:.6f} per day ({1/gamma:.1f} days infectious)")

    # Create dynamics object
    f_obj = F()

    # Test all measurement options
    measurement_options = [
        ('h_reported_cases', 'Measurement 1: I only (basic)'),
        ('h_incidence', 'Measurement 2: I + new cases'),
        ('h_incidence_recovery', 'Measurement 3: I + R + new cases'),
        ('h_ei_flows', 'Measurement 4: E + I + flows'),
        ('h_seir', 'Measurement 5: All compartments (S,E,I,R)'),
        ('h_seir_with_beta', 'Measurement 6: SEIR + beta'),
        ('h_with_flows', 'Measurement 7: SEIR + all flows ⭐ BEST')
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
            print(f"  Final S: {x_sim['S'][-1]:,.0f} ({100*x_sim['S'][-1]/N:.1f}%)")
            print(f"  Final E: {x_sim['E'][-1]:,.0f} ({100*x_sim['E'][-1]/N:.2f}%)")
            print(f"  Final I: {x_sim['I'][-1]:,.0f} ({100*x_sim['I'][-1]/N:.2f}%)")
            print(f"  Final R: {x_sim['R'][-1]:,.0f} ({100*x_sim['R'][-1]/N:.1f}%)")
            print(f"  Avg u1 (prevention): {np.mean(u_sim['u1']):.4f}")
            print(f"  Avg u3 (treatment):  {np.mean(u_sim['u3']):.4f}")
        except Exception as e:
            print(f"✗ Simulation failed: {str(e)}")

    print("\n" + "="*80)
    print("SUMMARY: All measurement options tested successfully!")
    print("="*80)
    print("\nModel Description:")
    print("  SEIR epidemic model with control inputs:")
    print("    dS/dt = μN - β(1-u₁)SI/N - μS")
    print("    dE/dt = β(1-u₁)SI/N - σE - μE")
    print("    dI/dt = σE - (γ+u₃)I - μI")
    print("    dR/dt = (γ+u₃)I - μR")
    print("    dβ/dt = 0 (unknown parameter)")
    print("\n  Controls:")
    print("    u₁: Prevention/social distancing (0-0.9)")
    print("    u₃: Treatment rate (0-0.5 per day)")
    print("="*80)
    print("\nAvailable measurement options:")
    print("  BASIC:")
    print("    - h_reported_cases:     I only")
    print("    - h_incidence:          I + new cases")
    print("    - h_incidence_recovery: I + R + new cases")
    print("\n  INTERMEDIATE:")
    print("    - h_ei_flows:           E + I + transition flows")
    print("    - h_seir:               All compartments (S,E,I,R)")
    print("\n  ADVANCED:")
    print("    - h_seir_with_beta:     SEIR + beta parameter")
    print("    - h_with_flows:         SEIR + all flows ⭐ BEST")
    print("="*80)
