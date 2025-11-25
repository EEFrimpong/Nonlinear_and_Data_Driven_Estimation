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
sigma_default = 1.0 / 5.2  # Progression rate from E to I (5.2 days incubation period)
gamma = 1.0 / 10.0         # Recovery rate (10 days infectious period)
N = 1_000_000              # Total population

# For this version, beta is a STATE, not seasonal
beta0_default = 0.3        # initial beta state value (can tune)

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, sigma=sigma_default, gamma=gamma, N=N,
                 beta0=beta0_default):
        """
        States:
            x = [S, E, I, R, beta]

        Controls:
            u1 = u_vec[0]   social distancing (transmission reduction)
            u2 = u_vec[1]   vaccination (S -> R)
            u3 = u_vec[2]   treatment (extra recovery of I)

        ODEs:
            dS/dt    = mu N - beta (1 - u1) S I / N - u2 S - mu S
            dE/dt    = beta (1 - u1) S I / N - sigma E - mu E
            dI/dt    = sigma E - (gamma + u3) I - mu I
            dR/dt    = (gamma + u3) I + u2 S - mu R
            dβ/dt    = 0
        """
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        self.beta0 = beta0

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous-time dynamics for the 5-state SEIR-beta model.
        """
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # Extract state variables
        S, E, I, R, beta = x_vec

        # Extract controls
        u1 = float(u_vec[0])   # prevention / social distancing
        u2 = float(u_vec[1])   # vaccination
        u3 = float(u_vec[2])   # treatment

        # Force of infection
        lambda_inf = beta * (1.0 - u1) * S * I / self.N

        # SEIR equations with controls (your model)
        dS_dt    = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt    = lambda_inf - self.sigma * E - self.mu * E
        dI_dt    = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt    = (self.gamma + u3) * I + u2 * S - self.mu * R

        # beta is constant in this version (can later add dynamics if you like)
        dbeta_dt = 0.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt])


############################################################################################
# Continuous time measurement functions (similar structure to your good code)
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, sigma=sigma_default,
                 gamma=gamma, N=N):
        """
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # -------------------------------------------------------------------------
    # 1. h_i_only: I
    # -------------------------------------------------------------------------
    def h_i_only(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I]
        """
        if return_measurement_names:
            return ['I_measured']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 2. h_ir: I, R
    # -------------------------------------------------------------------------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, R]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    # -------------------------------------------------------------------------
    # 3. h_reported_cases: I (reported infectious)
    # -------------------------------------------------------------------------
    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I]
        """
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 4. h_incidence: I, new_cases
    # new_cases = beta (1 - u1) S I / N
    # -------------------------------------------------------------------------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, new_cases]^T
        """
        if return_measurement_names:
            return ['I_reported', 'new_cases']

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
        """
        Measurement: y = [I, R, new_cases]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases']

        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, R, new_cases])

    # -------------------------------------------------------------------------
    # 6. h_ei_flows: E, I, new_inf, prog
    # new_inf = beta (1 - u1) S I / N
    # prog    = sigma E
    # -------------------------------------------------------------------------
    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [E, I, new_inf, prog]^T
        """
        if return_measurement_names:
            return ['E_measured', 'I_measured', 'new_inf', 'prog']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E

        return np.array([E, I, new_inf, prog])

    # -------------------------------------------------------------------------
    # 7. h_seir: S, E, I, R
    # -------------------------------------------------------------------------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R]^T
        """
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
        """
        Measurement: y = [S, E, I, R, new_inf]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        return np.array([S, E, I, R, new_inf])

    # -------------------------------------------------------------------------
    # 9. h_seir_with_beta: S, E, I, R, beta
    # -------------------------------------------------------------------------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R, beta]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'beta_measured']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        return np.array([S, E, I, R, beta])

    # -------------------------------------------------------------------------
    # 10. h_with_flows: S, E, I, R, new_inf, prog, recov
    # new_inf = beta (1 - u1) S I / N
    # prog    = sigma E
    # recov   = (gamma + u3) I
    # -------------------------------------------------------------------------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
            Measurement:
                y = [S, E, I, R, new_inf, prog, recov]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf', 'prog', 'recov']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        u1 = u_vec[0]
        u3 = u_vec[2]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E
        recov   = (self.gamma + u3) * I

        return np.array([S, E, I, R, new_inf, prog, recov])

    # -------------------------------------------------------------------------
    # 11. h_observable: I, R, new_cases, recov
    # (no C state in this 5-state version)
    # -------------------------------------------------------------------------
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, R, new_cases, recov]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases', 'recov']

        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        u1 = u_vec[0]
        u3 = u_vec[2]

        new_cases = beta * (1.0 - u1) * S * I / self.N
        recov     = (self.gamma + u3) * I

        return np.array([I, R, new_cases, recov])


############################################################################################
# SEIR simulation with MPC (same structure as your good code, but 5 states)
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
        beta0 = beta0_default

        x0 = np.array([S0, E0, I0, R0, beta0])

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

        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta': beta0_default * np.ones_like(tsim)
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

    # Bounds for beta
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.01
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
