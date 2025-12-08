import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
import pybounds

############################################################################################
# Set some global parameters
############################################################################################
mu = 0.02 / 365            # Natural mortality rate per day (2% per year)
sigma_default = 1.0 / 10.2  # Default progression rate from E to I (10.2 days incubation period)
gamma = 1.0 / 10.0         # Recovery rate (10 days infectious period)
N = 1_000_000              # Total population

# PARAMETERS FOR SEASONAL BETA_EFF
beta0_default = 0.5        # baseline transmission rate
T_default = 365            # seasonal period (days)

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, T=T_default):
        """Initialize with parameters stored as instance variables"""
        self.mu = mu
        self.gamma = gamma
        self.N = N

        # Seasonal beta parameters
        self.beta0 = beta0
        self.T = T

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control.

        States:
            x = [S, E, I, R, beta_eff, sigma, t]

        Controls:
            u1 = u_vec[0]   social distancing (transmission reduction)
            u2 = u_vec[1]   vaccination (S -> R)
            u3 = u_vec[2]   treatment (extra recovery of I)

        Seasonal transmission rate:
            beta_eff(t) = beta0 * sin(2π*t/T) * (1 - u1)

        Infection term (force of infection):
            lambda_inf = beta_eff * S * I / N

        ODEs:
            dS/dt          = μN - lambda_inf - u2*S - μ*S
            dE/dt          = lambda_inf - σ*E - μ*E
            dI/dt          = σ*E - (γ + u3)*I - μ*I
            dR/dt          = (γ + u3)*I + u2*S - μ*R
            d(beta_eff)/dt = beta0 * (2π/T) * cos(2πt/T) * (1-u1)
            dσ/dt          = 0
            dt/dt          = 1
        """

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_eff', 'sigma', 't']

        # Extract state variables
        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        t        = x_vec[6]

        # Extract controls
        u1 = u_vec[0]     # prevention / social distancing
        u2 = u_vec[1]     # vaccination
        u3 = u_vec[2]     # treatment

        # Force of infection using beta_eff state
        lambda_inf = beta_eff * S * I / self.N

        # SEIR equations with controls
        dS_dt = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - sigma * E - self.mu * E
        dI_dt = sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R

        # Dynamics for beta_eff (derivative of beta0 * sin(2πt/T) * (1-u1))
        # d/dt[beta0 * sin(2πt/T) * (1-u1)] = beta0 * (2π/T) * cos(2πt/T) * (1-u1)
        dbeta_dt = self.beta0 * (2.0 * np.pi / self.T) * np.cos(2.0 * np.pi * t / self.T) * (1.0 - u1)

        # Sigma treated as constant parameter-state
        dsigma_dt = 0.0
        
        # Time derivative
        dt_dt = 1.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt, dsigma_dt, dt_dt])


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, T=T_default):
        """
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N
        self.beta0 = beta0
        self.T = T

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # -------------------------------------------------------------------------
    # 1. h_i_only: I
    # -------------------------------------------------------------------------
    def h_i_only(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I]"""
        if return_measurement_names:
            return ['I_measured']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 2. h_ir: I, R
    # -------------------------------------------------------------------------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, R]^T"""
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    # -------------------------------------------------------------------------
    # 3. h_reported_cases: I
    # -------------------------------------------------------------------------
    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I]"""
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 4. h_incidence: I, new_cases
    # -------------------------------------------------------------------------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, new_cases]^T"""
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S        = x_vec[0]
        I        = x_vec[2]
        beta_eff = x_vec[4]

        new_cases = beta_eff * S * I / self.N
        return np.array([I, new_cases])

    # -------------------------------------------------------------------------
    # 5. h_ir_new: I, R, new_cases
    # -------------------------------------------------------------------------
    def h_ir_new(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, R, new_cases]^T"""
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases']

        S        = x_vec[0]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]

        new_cases = beta_eff * S * I / self.N
        return np.array([I, R, new_cases])

    # -------------------------------------------------------------------------
    # 6. h_ei_flows: E, I, new_inf, prog
    # -------------------------------------------------------------------------
    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [E, I, new_inf, prog]^T"""
        if return_measurement_names:
            return ['E_measured', 'I_measured', 'new_inf', 'prog']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]

        new_inf = beta_eff * S * I / self.N
        prog = sigma * E

        return np.array([E, I, new_inf, prog])

    # -------------------------------------------------------------------------
    # 7. h_seir: S, E, I, R
    # -------------------------------------------------------------------------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]

        return np.array([S, E, I, R])

    # -------------------------------------------------------------------------
    # 8. h_seir_flows: S, E, I, R, new_inf
    # -------------------------------------------------------------------------
    def h_seir_flows(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, new_inf]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'new_inf']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]

        new_inf = beta_eff * S * I / self.N
        return np.array([S, E, I, R, new_inf])

    # -------------------------------------------------------------------------
    # 9. h_seir_with_beta: S, E, I, R, beta_eff
    # -------------------------------------------------------------------------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, beta_eff]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_eff']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]

        return np.array([S, E, I, R, beta_eff])

    # -------------------------------------------------------------------------
    # 10. h_with_flows: S, E, I, R, new_inf, prog, recov
    # -------------------------------------------------------------------------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, new_inf, prog, recov]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf', 'prog', 'recov']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        u3       = u_vec[2]

        new_inf = beta_eff * S * I / self.N
        prog = sigma * E
        recov = (self.gamma + u3) * I

        return np.array([S, E, I, R, new_inf, prog, recov])

    # -------------------------------------------------------------------------
    # 11. h_observable: I, R, new_cases, recov
    # -------------------------------------------------------------------------
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, R, new_cases, recov]^T"""
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases', 'recov']

        S        = x_vec[0]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        u3       = u_vec[2]
        
        recov = (self.gamma + u3) * I
        new_cases = beta_eff * S * I / self.N

        return np.array([I, R, new_cases, recov])


############################################################################################
# SEIR simulation with MPC
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):

    # Default initial conditions
    if x0 is None:
        S0 = 0.70 * N
        E0 = 0.05 * N
        I0 = 0.05 * N
        R0 = N - S0 - E0 - I0
        t0 = 0.0
        
        # Initial beta_eff consistent with seasonal formula at t=0
        # beta_eff(0) = beta0 * sin(0) * (1 - 0) = 0
        beta_eff0 = beta0_default * np.sin(0.0)
        sigma0 = sigma_default

        x0 = np.array([S0, E0, I0, R0, beta_eff0, sigma0, t0])

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

    # Default setpoint (only on E and I; others zero or nominal)
    if setpoint is None:
        I_initial = x0[2]
        E_initial = x0[1]

        I_target = 0.0001 * N
        E_target = 0.00005 * N

        I_set = I_target + (I_initial - I_target) * np.exp(-tsim / 100.0)
        E_set = E_target + (E_initial - E_target) * np.exp(-tsim / 80.0)

        # Expected beta_eff trajectory based on seasonal formula
        beta_eff_expected = beta0_default * np.sin(2.0 * np.pi * tsim / T_default)
        # Ensure non-negative
        beta_eff_expected = np.maximum(0.0, beta_eff_expected)

        # tvp keys become X_set internally in pybounds
        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta_eff': beta_eff_expected,
            'sigma': sigma_default * np.ones_like(tsim),
            't': tsim  # Time state tracks simulation time
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost (penalize E and I only)
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

    # Bounds for beta_eff
    simulator.mpc.bounds['lower', '_x', 'beta_eff'] = 0.0
    simulator.mpc.bounds['upper', '_x', 'beta_eff'] = 2.0
    
    # Bounds for sigma
    simulator.mpc.bounds['lower', '_x', 'sigma'] = 1.0 / 30.0   # incubation up to 30 days
    simulator.mpc.bounds['upper', '_x', 'sigma'] = 1.0          # up to 1/day
    
    # Bounds for time state
    simulator.mpc.bounds['lower', '_x', 't'] = 0.0
    simulator.mpc.bounds['upper', '_x', 't'] = tsim_length + 100.0

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
