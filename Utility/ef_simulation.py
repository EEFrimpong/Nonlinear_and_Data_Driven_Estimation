import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
# NOTE: The pybounds library is assumed to be available for the simulator to run.
# import pybounds

############################################################################################
# Set some global parameters
############################################################################################
mu = 0.02 / 365            # Natural mortality rate per day (2% per year)
sigma_default = 1.0 / 10.2 # Progression rate from E to I (constant)
gamma = 1.0 / 10.0         # Recovery rate (10 days infectious period)
N = 1_000_000              # Total population

# PARAMETERS FOR BETA (Simplified, as beta is now a state, but dbeta_dt = 0.0)
beta0_default = 0.5        # Baseline transmission value for initialization
# epsilon_default = 0.2    # Removed seasonal parameters
# T_default = 365

# ---
# NOTE: The control vector u_vec is assumed to be u = [u1, u2, u3]
# u1: social distancing (transmission reduction)
# u2: vaccination (S -> R)
# u3: treatment (extra recovery of I)

############################################################################################
# Continuous time dynamics function (F Class)
############################################################################################
class F(object):
    def __init__(self, mu=mu, gamma=gamma, N=N, sigma=sigma_default):
        """Initialize with parameters stored as instance variables"""
        self.mu = mu
        self.gamma = gamma
        self.N = N
        self.sigma = sigma 

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for the 5-state SEIR model with control.

        States: x = [S, E, I, R, beta] (Size 5)
        Controls: u = [u1, u2, u3]
        """

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # Extract state variables (5 states)
        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        # Extract controls
        u1 = u_vec[0]
        u2 = u_vec[1]
        u3 = u_vec[2]

        # Force of infection
        lambda_inf = beta * (1.0 - u1) * S * I / self.N

        # ODEs (using self.sigma for progression rate)
        dS_dt = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - self.sigma * E - self.mu * E
        dI_dt = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R
        dbeta_dt = 0.0     # beta is treated as a constant parameter state

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt])

# ---

############################################################################################
# Continuous time measurement functions (H Class)
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N, sigma=sigma_default):
        """
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N
        self.sigma = sigma 

    def h(self, x_vec, u_vec, return_measurement_names=False):
        """
        Main measurement function dispatcher.
        """
        if not hasattr(self, self.measurement_option):
             raise AttributeError(f"Measurement option '{self.measurement_option}' not found.")
             
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # Note: All following functions use the 5-state vector [S, E, I, R, beta]

    # -------------------------------------------------------------------------
    # 1. h_i_only: I
    # -------------------------------------------------------------------------
    def h_i_only(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_measured']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 2. h_ir: I, R
    # -------------------------------------------------------------------------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_measured', 'R_measured']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    # -------------------------------------------------------------------------
    # 3. h_reported_cases: I
    # -------------------------------------------------------------------------
    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 4. h_incidence: I, new_cases = beta * (1 - u1) S I / N
    # -------------------------------------------------------------------------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_reported', 'new_cases']
        S    = x_vec[0]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]
        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, new_cases])

    # -------------------------------------------------------------------------
    # 5. h_incidence_recovery: I, R, new_cases
    # -------------------------------------------------------------------------
    def h_incidence_recovery(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_measured', 'R_measured', 'new_cases']
        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1   = u_vec[0]
        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, R, new_cases])

    # -------------------------------------------------------------------------
    # 6. h_ei_flows: E, I, new_inf, prog = sigma * E
    # -------------------------------------------------------------------------
    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['E_measured', 'I_measured', 'new_inf', 'prog']
        S     = x_vec[0]
        E     = x_vec[1]
        I     = x_vec[2]
        beta  = x_vec[4]
        u1    = u_vec[0]
        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E # Uses constant sigma
        return np.array([E, I, new_inf, prog])

    # -------------------------------------------------------------------------
    # 7. h_seir: S, E, I, R
    # -------------------------------------------------------------------------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['S_measured', 'E_measured', 'I_measured', 'R_measured']
        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        return np.array([S, E, I, R])

    # -------------------------------------------------------------------------
    # 8. h_seir_flows: S, E, I, R, new_inf
    # -------------------------------------------------------------------------
    def h_seir_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'new_inf']
        S     = x_vec[0]
        E     = x_vec[1]
        I     = x_vec[2]
        R     = x_vec[3]
        beta  = x_vec[4]
        u1    = u_vec[0]
        new_inf = beta * (1.0 - u1) * S * I / self.N
        return np.array([S, E, I, R, new_inf])

    # -------------------------------------------------------------------------
    # 9. h_seir_with_beta: S, E, I, R, beta
    # -------------------------------------------------------------------------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_measured']
        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        return np.array([S, E, I, R, beta])

    # -------------------------------------------------------------------------
    # 10. h_with_flows: S, E, I, R, new_inf, prog, recov = (gamma + u3) I
    # -------------------------------------------------------------------------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'new_inf', 'prog', 'recov']
        S     = x_vec[0]
        E     = x_vec[1]
        I     = x_vec[2]
        R     = x_vec[3]
        beta  = x_vec[4]
        u1    = u_vec[0]
        u3    = u_vec[2]
        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E
        recov   = (self.gamma + u3) * I
        return np.array([S, E, I, R, new_inf, prog, recov])

    # -------------------------------------------------------------------------
    # 11. h_observable: I, R, new_cases, recov
    # -------------------------------------------------------------------------
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_measured', 'R_measured', 'new_cases', 'recov']
        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]
        u3 = u_vec[2]
        recov     = (self.gamma + u3) * I
        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, R, new_cases, recov])

# ---

############################################################################################
# SEIR simulation function (Adapted for 5 states)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):

    # Default initial conditions for [S, E, I, R, beta]
    if x0 is None:
        S0 = 0.70 * N
        E0 = 0.05 * N
        I0 = 0.05 * N
        R0 = N - S0 - E0 - I0
        beta0 = beta0_default # Use the global default for initialization

        # 5 states: S, E, I, R, beta
        x0 = np.array([S0, E0, I0, R0, beta0]) 

    # State and input names
    state_names = f(None, None, return_state_names=True) 
    input_names = ['u1', 'u2', 'u3']

    # Measurement names
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # NOTE: Uncomment and ensure 'pybounds' is installed to run the simulation setup below.
    """
    # Initialize simulator
    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10/dt)
    )

    # Add measurement noise if provided
    if measurement_noise_stds is not None:
        noise_std_array = []
        for meas in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(meas, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)

    # Time grid
    tsim = np.arange(0, tsim_length, step=dt)

    # Default setpoint
    if setpoint is None:
        I_initial = x0[2]
        E_initial = x0[1]
        
        I_target = 0.0001 * N
        E_target = 0.00005 * N

        I_set = I_target + (I_initial - I_target) * np.exp(-tsim / 100.0)
        E_set = E_target + (E_initial - E_target) * np.exp(-tsim / 80.0)
        
        # tvp keys become X_set internally in pybounds
        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta': x0[4] * np.ones_like(tsim) # Target beta is constant
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost (penalize E and I only)
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2
    cost = 10.0 * cost_E + 100.0 * cost_I

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Input penalties
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3)

    # State bounds (Adapted for 5 states)
    eps = 1e-6
    simulator.mpc.bounds['lower', '_x', 'S'] = eps
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = eps
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = eps
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = eps
    simulator.mpc.bounds['upper', '_x', 'R'] = N
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'beta'] = 2.0 

    # Control bounds
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.6
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.5

    # Run simulation with MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator
    """
    # Placeholder return if not using pybounds:
    tsim = np.arange(0, tsim_length, step=dt)
    return tsim, np.zeros((len(tsim), 5)), np.zeros((len(tsim), 3)), np.zeros((len(tsim), len(measurement_names))), None
