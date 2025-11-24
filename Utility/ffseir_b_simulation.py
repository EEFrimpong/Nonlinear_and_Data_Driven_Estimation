# ===============================================================
# FULL SEASONAL-B SEIR MODEL WITH MULTIPLE MEASUREMENTS + MPC
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import pybounds

###############################################################
# GLOBAL PARAMETERS
###############################################################
mu = 0.02 / 365
sigma = 1 / 5.2
gamma = 1 / 10
N = 10_000_000

beta0_default    = 0.3
epsilon_default  = 0.01
T_default        = 365


###############################################################
# 1. SEIR DYNAMICS: x = [S, E, I, R, beta]
#    beta acts as a "clock": d(beta)/dt = 1
###############################################################
class F(object):
    def __init__(self,
                 mu=mu,
                 sigma=sigma,
                 gamma=gamma,
                 N=N,
                 beta0=beta0_default,
                 epsilon=epsilon_default,
                 T=T_default):

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
        """Dynamics f(x,u) with seasonal beta & 2 controls u1, u2."""

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # unpack state and inputs
        S, E, I, R, beta_phase = x_vec
        u1, u2 = u_vec    # e.g. u1: distancing, u2: vaccination/treatment

        # seasonal forcing: use beta_phase as a "time/phase" variable
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * beta_phase / self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)

        # force of infection
        lam = beta_eff * S * I / self.N

        # SEIR equations
        dS = self.mu*self.N - lam - self.mu*S - u2*S
        dE = lam - self.sigma*E - self.mu*E
        dI = self.sigma*E - self.gamma*I - self.mu*I
        dR = self.gamma*I + u2*S - self.mu*R

        # beta_phase evolves at unit speed -> acts like time
        dBeta = 1.0

        return np.array([dS, dE, dI, dR, dBeta])


###############################################################
# 2. MEASUREMENT CLASS (H) WITH MANY OPTIONS
###############################################################
class H(object):
    def __init__(self,
                 measurement_option,
                 mu=mu,
                 sigma=sigma,
                 gamma=gamma,
                 N=N,
                 beta0=beta0_default,
                 epsilon=epsilon_default,
                 T=T_default):
        """
        measurement_option: name of the measurement function to use, e.g.
            'h_ir',
            'h_reported_cases',
            'h_incidence',
            'h_seir',
            'h_ir_newcases',
            'h_seir_with_beta',
            'h_with_flows'
        """
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
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # ---------- helper to reconstruct beta_eff consistent with F ----------
    def _beta_eff(self, x_vec, u_vec):
        S, E, I, R, beta_phase = x_vec
        u1, u2 = u_vec
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * beta_phase / self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)
        return beta_eff

    # ---------- basic: I and R ----------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        S, E, I, R, beta_phase = x_vec
        return np.array([I, R])

    # ---------- reported cases: I only ----------
    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported']
        S, E, I, R, beta_phase = x_vec
        return np.array([I])

    # ---------- incidence: I + new cases (lambda) ----------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S, E, I, R, beta_phase = x_vec
        beta_eff = self._beta_eff(x_vec, u_vec)
        new_cases = beta_eff * S * I / self.N

        return np.array([I, new_cases])

    # ---------- full SEIR (no beta) ----------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured']
        S, E, I, R, beta_phase = x_vec
        return np.array([S, E, I, R])

    # ---------- I, R, and new infections ----------
    def h_ir_newcases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_infections']

        S, E, I, R, beta_phase = x_vec
        beta_eff = self._beta_eff(x_vec, u_vec)
        new_inf = beta_eff * S * I / self.N

        return np.array([I, R, new_inf])

    # ---------- SEIR plus beta_phase ----------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_phase']
        S, E, I, R, beta_phase = x_vec
        return np.array([S, E, I, R, beta_phase])

    # ---------- SEIR plus flows (new infections, progressions, recoveries) ----------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_infections', 'progressions', 'recoveries']

        S, E, I, R, beta_phase = x_vec
        u1, u2 = u_vec

        beta_eff = self._beta_eff(x_vec, u_vec)
        new_inf = beta_eff * S * I / self.N
        prog    = self.sigma * E
        rec     = self.gamma * I

        return np.array([S, E, I, R, new_inf, prog, rec])


###############################################################
# 3. SIMULATE WITH MPC (2 controls: u1, u2)
###############################################################
def simulate_seir(f,
                  h,
                  tsim_length=365,
                  dt=1.0,
                  measurement_names=None,
                  setpoint=None,
                  rterm_u1=1e-4,
                  rterm_u2=1e-4,
                  x0=None,
                  measurement_noise_stds=None):
    """
    MPC-compatible SEIR simulation.

    f : callable
        Dynamics function: f(x_vec, u_vec, return_state_names=False)
        (can be F().f or F() directly thanks to __call__).
    h : callable
        Measurement function: h(x_vec, u_vec, return_measurement_names=False)
    """

    # -----------------------------------------------------------
    # State and input names
    # -----------------------------------------------------------
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2']

    # -----------------------------------------------------------
    # Measurement names
    # -----------------------------------------------------------
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # -----------------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------------
    if x0 is None:
        S0    = 0.90 * N
        E0    = 0.05 * N
        I0    = 0.04 * N
        R0    = 0.0
        beta0 = 0.0          # start clock at phase 0
        x0    = np.array([S0, E0, I0, R0, beta0])

    # -----------------------------------------------------------
    # Build simulator
    # -----------------------------------------------------------
    simulator = pybounds.Simulator(
        f,
        h,
        dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10 / dt)
    )

    # Optional measurement noise
    if measurement_noise_stds is not None:
        noise_std_array = [measurement_noise_stds.get(name, 0.0)
                           for name in measurement_names]
        simulator.measurement_noise_std = np.array(noise_std_array)

    # -----------------------------------------------------------
    # Time grid
    # -----------------------------------------------------------
    tsim = np.arange(0, tsim_length, step=dt)

    # -----------------------------------------------------------
    # Setpoint (TVPs)
    #   Important: tvp keys for MPC are 'E_set' and 'I_set'
    # -----------------------------------------------------------
    if setpoint is None:
        I_set = 0.0001 * N + (x0[2] - 0.0001 * N) * np.exp(-tsim / 100.0)
        E_set = 0.00005 * N + (x0[1] - 0.00005 * N) * np.exp(-tsim / 80.0)

        setpoint = {
            'E_set': E_set,
            'I_set': I_set
        }

    setpoint_processed = {}
    for key, arr in setpoint.items():
        arr = np.asarray(arr).squeeze()
        if arr.ndim == 0:
            arr = np.ones_like(tsim) * float(arr)
        else:
            arr = np.resize(arr, tsim.shape[0])
        setpoint_processed[key] = arr

    simulator.update_dict(setpoint_processed, name='setpoint')

    # -----------------------------------------------------------
    # Cost: penalize E and I deviations from setpoints
    # -----------------------------------------------------------
    cost = 0
    if 'E' in state_names:
        cost += 10 * (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    if 'I' in state_names:
        cost += 100 * (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2)

    # -----------------------------------------------------------
    # Bounds
    # -----------------------------------------------------------
    eps = 1e-6
    for var in state_names:
        if var == 'beta':
            simulator.mpc.bounds['lower', '_x', 'beta'] = -1e6  # clock can grow
            simulator.mpc.bounds['upper', '_x', 'beta'] =  1e6
        else:
            simulator.mpc.bounds['lower', '_x', var] = eps
            simulator.mpc.bounds['upper', '_x', var] = N

    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9

    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.5

    # -----------------------------------------------------------
    # Simulate with MPC
    # -----------------------------------------------------------
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


###############################################################
# 4. QUICK SELF-TEST
###############################################################
if __name__ == "__main__":
    f = F()                    # dynamics
    h = H('h_with_flows')      # try any of:
                               # 'h_ir', 'h_reported_cases', 'h_incidence',
                               # 'h_seir', 'h_ir_newcases',
                               # 'h_seir_with_beta', 'h_with_flows'

    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
        f,
        h,
        tsim_length=365,
        dt=1.0
    )

    # example: plot I(t)
    plt.figure()
    plt.plot(t_sim, x_sim['I'])
    plt.xlabel("Time (days)")
    plt.ylabel("Infectious I(t)")
    plt.title("SEIR Seasonal-beta with MPC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

