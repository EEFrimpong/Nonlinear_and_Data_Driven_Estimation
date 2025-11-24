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
N = 10000000          # Total population

# NEW PARAMETERS FOR SEASONAL BETA
beta0_default = 0.5    # baseline transmission
epsilon_default = 0.2  # seasonal amplitude
T_default = 365        # seasonal period

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, sigma=sigma, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):
        """Initialize with parameters stored as instance variables"""
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

        # NEW seasonal parameters
        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control.

        NEW ADDITION:
            β_eff(t) = β0 (1 + ε cos(2πt/T)) (1 - u1)

        NOTE:
            Because pybounds does NOT pass time into f(),
            we pass time as u_vec[2].

        State vector (modified):
            x = [S, E, I, R, beta_dummy]

        Controls:
            u1 = u_vec[0]   social distancing
            u2 = u_vec[1]   treatment
            u3  = u_vec[2]   (time injected manually)
        """

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_dummy']

        # Extract state variables
        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta_dummy = x_vec[4]  # no longer used, kept for compatibility

        # Extract controls
        u1 = u_vec[0]     # prevention
        u2 = u_vec[1]     # treatment
        u3 = u_vec[2]      # TIME (important!!)

        # Compute β_eff(t)
        seasonal = 1.0 + self.epsilon * np.cos(2 * np.pi * t / self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)

        # Force of infection
        lambda_inf = beta_eff * S * I / self.N

        # SEIR equations
        dS_dt = self.mu * self.N - lambda_inf u2*S- self.mu * S
        dE_dt = lambda_inf - self.sigma * E - self.mu * E
        dI_dt = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2*S- self.mu * R

        # beta_dummy remains constant (still part of state vector for compatibility)
        dbeta_dt = 0.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt])


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, sigma=sigma, gamma=gamma, N=N):
        self.measurement_option = measurement_option
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported', 'new_cases']
        S = x_vec[0]
        I = x_vec[2]
        beta_eff = x_vec[4]  # dummy, not used
        u1 = u_vec[0]
        # incidence approximated from beta_dummy for compatibility
        new_cases = beta_eff * (1 - u1) * S * I / self.N
        return np.array([I, new_cases])

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured']
        return np.array([x_vec[0], x_vec[1], x_vec[2], x_vec[3]])

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        return np.array([ x_vec[2], x_vec[3]])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta']
        return np.array([x_vec[0], x_vec[1], x_vec[2], x_vec[3], x_vec[4]])

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured','E_measured','I_measured','R_measured',
                    'new_infections','progressions','recoveries']

        S,E,I,R,beta_dummy = x_vec
        u1 = u_vec[0]
        u2 = u_vec[1]
        u3 = u_vec[2]

        # flows approximated with dummy beta for compatibility
        new_inf = beta_dummy*(1-u1)*S*I/self.N
        prog = self.sigma * E
        rec = (self.gamma + u3) * I

        return np.array([S,E,I,R,new_inf,prog,rec])


############################################################################################
# SEIR simulation with MPC (MODIFIED TO PASS TIME INTO u_vec)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2 = 1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):

    # Default initial conditions
    if x0 is None:
        S0 = 0.90 * N
        E0 = 0.01 * N
        I0 = 0.01 * N
        R0 = N - S0 - E0 - I0
        beta0 = beta0_default
        x0 = np.array([S0, E0, I0, R0, beta0])

    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']   # <<< NEW THIRD INPUT = TIME

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10/dt)
    )

    # Set measurement noise
    if measurement_noise_stds is not None:
        noise_std_array = []
        for meas in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(meas, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)

    tsim = np.arange(0, tsim_length, step=dt)

    # Set default setpoint
    if setpoint is None:
        I_set = 0.0001*N + (x0[2]-0.0001*N)*np.exp(-tsim/100)
        E_set = 0.00005*N + (x0[1]-0.00005*N)*np.exp(-tsim/80)
        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta_dummy': np.ones_like(tsim)*beta0_default
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2
    cost = 10*cost_E + 100*cost_I

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3)

    # State bounds
    eps = 1e-6
    for var in ['S','E','I','R']:
        simulator.mpc.bounds['lower','_x',var] = eps
        simulator.mpc.bounds['upper','_x',var] = N
    simulator.mpc.bounds['lower','_x','beta_dummy'] = 0.1
    simulator.mpc.bounds['upper','_x','beta_dummy'] = 2.0

    # Control bounds
    simulator.mpc.bounds['lower','_u','u1'] = 0.0
    simulator.mpc.bounds['upper','_u','u1'] = 0.9
    simulator.mpc.bounds['lower','_u','u3'] = 0.0
    simulator.mpc.bounds['upper','_u','u3'] = 0.5
    simulator.mpc.bounds['lower','_u','u2'] = 0.0
    simulator.mpc.bounds['upper','_u','u2'] = 0.6

    # Run simulation (NOTE: simulator automatically supplies time to f)
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator
