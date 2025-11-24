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
sigma_default = 1.0 / 5.2    # Default progression rate from E to I
gamma = 1.0 / 10.0   # Recovery rate (10 days infectious period)
N = 10000000          # Total population

# PARAMETERS FOR BETA_EFF (used as initial guess / prior)
beta0_default = 0.5    # baseline transmission (initial beta_eff)
epsilon_default = 0.2  # kept for reference if you later want seasonal structure
T_default = 365        # kept for reference

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, gamma=gamma, N=N):
        """Initialize with parameters stored as instance variables."""
        self.mu = mu
        self.gamma = gamma
        self.N = N

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control,
        with beta_eff and sigma treated as STATES.

        State vector:
            x = [S, E, I, R, beta_eff, sigma]

        Controls:
            u_vec[0] = u1  (social distancing / transmission reduction)
            u_vec[1] = u2  (vaccination rate S -> R)
            u_vec[2] = u3  (treatment rate I -> R)

        Dynamics (conceptually):

            dS/dt = μN - beta_eff * (1 - u1) * S I / N - u2 S - μS
            dE/dt = beta_eff * (1 - u1) * S I / N - σ E - μE
            dI/dt = σ E - (γ + u3) I - μI
            dR/dt = (γ + u3) I + u2 S - μR
            d(beta_eff)/dt = 0
            dσ/dt = 0
        """

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_eff', 'sigma']

        # Extract state variables
        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]

        # Extract control inputs
        u1 = u_vec[0]   # prevention/social distancing
        u2 = u_vec[1]   # vaccination rate
        u3 = u_vec[2]   # treatment rate

        # Force of infection with beta_eff and control u1
        lambda_inf = beta_eff * (1.0 - u1) * S * I / self.N

        # SEIR dynamics with controls and sigma as a state
        dS_dt = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - sigma * E - self.mu * E
        dI_dt = sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R

        # Parameter-states: constant in time (to be estimated/observed)
        dbeta_dt  = 0.0
        dsigma_dt = 0.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt, dsigma_dt])


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N):
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I]
        """
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, new_cases]^T,
        where new_cases is the infection flow beta_eff (1 - u1) S I / N.
        """
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S        = x_vec[0]
        I        = x_vec[2]
        beta_eff = x_vec[4]
        u1       = u_vec[0]

        new_cases = beta_eff * (1.0 - u1) * S * I / self.N
        return np.array([I, new_cases])

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured']
        return np.array([x_vec[0], x_vec[1], x_vec[2], x_vec[3]])

    def h_sir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, I, R]^T
        """
        if return_measurement_names:
            return ['S_measured, 'I_measured', 'R_measured']
        return np.array([x_vec[0], x_vec[2], x_vec[3]])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R, beta_eff, sigma]^T
        Full compartments plus parameter-states.
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'beta_eff', 'sigma']
        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        return np.array([S, E, I, R, beta_eff, sigma])

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S,E,I,R,new_infections,progressions,recoveries]^T
        - new_infections = beta_eff (1 - u1) S I / N
        - progressions   = sigma E
        - recoveries     = (gamma + u3) I
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_infections', 'progressions', 'recoveries']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]

        u1 = u_vec[0]
        u3 = u_vec[2]

        new_inf    = beta_eff * (1.0 - u1) * S * I / self.N
        progress   = sigma * E
        recoveries = (self.gamma + u3) * I

        return np.array([S, E, I, R, new_inf, progress, recoveries])


############################################################################################
# SEIR simulation with MPC (no time injected into u_vec)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):
    """
    Simulate SEIR disease model with MPC control.

    States:
        [S, E, I, R, beta_eff, sigma]

    Controls:
        u1: prevention/social distancing
        u2: vaccination
        u3: treatment
    """

    # Default initial conditions
    if x0 is None:
        S0 = 0.90 * N
        E0 = 0.01 * N
        I0 = 0.01 * N
        R0 = N - S0 - E0 - I0
        beta_eff0 = beta0_default      # initial guess for beta_eff
        sigma0    = sigma_default      # initial guess for sigma

        x0 = np.array([S0, E0, I0, R0, beta_eff0, sigma0])

    # State and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']

    # Measurement names
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

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

        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta_eff': beta0_default * np.ones_like(tsim),
            'sigma': sigma_default * np.ones_like(tsim)
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost: penalize E and I deviations from setpoint
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2
    cost = 10.0 * cost_E + 100.0 * cost_I

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Input penalties
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3)

    # State bounds
    eps = 1e-6
    simulator.mpc.bounds['lower', '_x', 'S'] = eps
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = eps
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = eps
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = eps
    simulator.mpc.bounds['upper', '_x', 'R'] = N

    # Bounds for beta_eff and sigma as states
    simulator.mpc.bounds['lower', '_x', 'beta_eff'] = 0.1
    simulator.mpc.bounds['upper', '_x', 'beta_eff'] = 2.0
    simulator.mpc.bounds['lower', '_x', 'sigma'] = 1/30.0   # e.g. incubation up to 30 days
    simulator.mpc.bounds['upper', '_x', 'sigma'] = 1.0      # up to 1/day

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
