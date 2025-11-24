import numpy as np
import matplotlib.pyplot as plt
import pybounds

############################################################################################
# GLOBAL PARAMETERS
############################################################################################
mu = 0.02 / 365
sigma = 1.0 / 5.2
gamma = 1.0 / 10.0
N = 10_000_000

beta0_default = 0.5
epsilon_default = 0.2
T_default = 365


############################################################################################
# DYNAMICS (F CLASS)
############################################################################################
class F(object):

    def __init__(self, mu=mu, sigma=sigma, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):

        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def __call__(self, x_vec, u_vec, return_state_names=False):
        return self.f(x_vec, u_vec, return_state_names)

    def f(self, x_vec, u_vec, return_state_names=False):

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_dummy']

        S, E, I, R, beta_dummy = x_vec
        u1, u3, t = u_vec

        # seasonal beta
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * t / self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)

        # force of infection
        lambda_inf = beta_eff * S * I / self.N

        dS = self.mu * self.N - lambda_inf - self.mu * S
        dE = lambda_inf - self.sigma * E - self.mu * E
        dI = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR = (self.gamma + u3) * I - self.mu * R

        dBeta = 0.0
        return np.array([dS, dE, dI, dR, dBeta])


############################################################################################
# MEASUREMENT CLASS (H)
############################################################################################
class H(object):

    def __init__(self, measurement_option, mu=mu, sigma=sigma, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):

        self.measurement_option = measurement_option
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def __call__(self, x_vec, u_vec, return_measurement_names=False):
        return self.h(x_vec, u_vec, return_measurement_names)

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = getattr(self, self.measurement_option)
        return func(x_vec, u_vec, return_measurement_names)

    # helper
    def _beta_eff_from_u(self, x_vec, u_vec):
        u1, u3, t = u_vec
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * t / self.T)
        return self.beta0 * seasonal * (1 - u1)

    ######################## measurements #########################

    def h_ir_newcases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_infections']

        S, E, I, R, b = x_vec
        beta_eff = self._beta_eff_from_u(x_vec, u_vec)
        new_inf = beta_eff * S * I / self.N
        return np.array([I, R, new_inf])

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported']
        return np.array([x_vec[2]])

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported', 'new_cases']
        S, E, I, R, b = x_vec
        beta_eff = self._beta_eff_from_u(x_vec, u_vec)
        return np.array([I, beta_eff * S * I / self.N])

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S','E','I','R']
        return np.array(x_vec[:4])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S','E','I','R','beta_dummy']
        return np.array(x_vec)

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S','E','I','R','new_infections','progressions','recoveries']
        S, E, I, R, _ = x_vec
        u1, u3, t = u_vec
        beta_eff = self._beta_eff_from_u(x_vec, u_vec)
        new_inf = beta_eff * S * I / self.N
        prog = self.sigma * E
        rec = (self.gamma + u3) * I
        return np.array([S, E, I, R, new_inf, prog, rec])


############################################################################################
# SIMULATOR WITH CORRECT TVP HANDLING (NO S_set, NO R_set)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0,
                  measurement_names=None, setpoint=None,
                  rterm_u1=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):

    # INITIAL CONDITIONS
    if x0 is None:
        S0 = 0.90*N
        E0 = 0.01*N
        I0 = 0.01*N
        R0 = N - S0 - E0 - I0
        beta0 = beta0_default
        x0 = np.array([S0, E0, I0, R0, beta0])

    # names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1','u3','time']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # BUILD SIMULATOR
    simulator = pybounds.Simulator(
        f,h,
        dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10/dt)
    )

    # NOISE
    if measurement_noise_stds is not None:
        simulator.measurement_noise_std = np.array(
            [measurement_noise_stds.get(m,0.0) for m in measurement_names]
        )

    tsim = np.arange(0,tsim_length,dt)

    # ✔ ONLY USE VALID TVPs: E_set, I_set
    if setpoint is None:
        E_set = 0.00005*N + (x0[1] - 0.00005*N)*np.exp(-tsim/80)
        I_set = 0.0001*N  + (x0[2] - 0.0001*N)*np.exp(-tsim/100)
        setpoint = {'E_set':E_set, 'I_set':I_set}

    # update TVPs
    simulator.update_dict(setpoint, name='setpoint')

    # COST
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2
    cost = 10*cost_E + 100*cost_I

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=rterm_u1, u3=rterm_u3)

    # BOUNDS
    eps = 1e-6
    for var in ['S','E','I','R']:
        simulator.mpc.bounds['lower','_x',var] = eps
        simulator.mpc.bounds['upper','_x',var] = N
    simulator.mpc.bounds['lower','_x','beta_dummy'] = 0.1
    simulator.mpc.bounds['upper','_x','beta_dummy'] = 2.0

    simulator.mpc.bounds['lower','_u','u1'] = 0.0
    simulator.mpc.bounds['upper','_u','u1'] = 0.9

    simulator.mpc.bounds['lower','_u','u3'] = 0.0
    simulator.mpc.bounds['upper','_u','u3'] = 0.5

    simulator.mpc.bounds['lower','_u','time'] = 0.0
    simulator.mpc.bounds['upper','_u','time'] = tsim_length

    #  RUN SIMULATION
    return simulator.simulate(
        x0=x0, u=None, mpc=True, return_full_output=True
    )
