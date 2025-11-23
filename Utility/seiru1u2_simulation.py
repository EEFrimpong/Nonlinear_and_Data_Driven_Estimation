import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate

import pybounds

###############################################################################
# Global SEIR parameters
###############################################################################
mu    = 0.02 / 365      # Natural mortality rate per day (2% per year)
sigma = 1.0 / 5.2       # Progression rate from E to I (5.2 days incubation period)
gamma = 1.0 / 10.0      # Recovery rate (10 days infectious period)
N     = 10_000_000       # Total population

###############################################################################
# Dynamics: F class
###############################################################################
class F(object):
    """
    SEIR model with control and beta as a state:
        x = [S, E, I, R, beta]

    Controls:
        u = [u1, u2, u3]
        u1: prevention/social distancing (0–1, reduces transmission)
        u2: vaccination
        u3: treatment rate (adds to gamma)
    """
    def __init__(self, mu=mu, sigma=sigma, gamma=gamma, N=N):
        self.mu    = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N     = N

    def f(self, x_vec, u_vec, return_state_names=False):
        if return_state_names:
            # Must match len(x_vec)
            return ['S', 'E', 'I', 'R', 'beta']

        # unpack state
        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        # controls
        u1 = u_vec[0]  # prevention/social distancing
        u2 = u_vec[1]  # treatment
        
        # incidence term
        lambda_inf = beta * (1 - u1) * S * I / self.N

        dSdt    = self.mu * self.N - lambda_inf -u2 * S- self.mu * S
        dEdt    = lambda_inf - self.sigma * E - self.mu * E
        dIdt    = self.sigma * E - (self.gamma) * I - self.mu * I
        dRdt    = (self.gamma) * I + u2 * S- self.mu * R
        dbetadt = 0.0  # constant beta as state

        return np.array([dSdt, dEdt, dIdt, dRdt, dbetadt])

###############################################################################
# Measurement: H class
###############################################################################
class H(object):
    """
    Measurement wrapper. Use as:

        h_inc = H('h_incidence').h
        y = h_inc(x, u)

    Implemented measurements:
        - h_incidence
        - h_seir
        - h_ir
        - h_i
        - h_incidence_recovery
    """
    def __init__(self, measurement_option, mu=mu, sigma=sigma, gamma=gamma, N=N):
        self.measurement_option = measurement_option
        self.mu    = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N     = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # ---------------------------------------------------------------------
    # h_i: just I(t)
    # ---------------------------------------------------------------------
    def h_i(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I']

        I = x_vec[2]
        return np.array([I])

    # ---------------------------------------------------------------------
    # h_ir: I(t), R(t)
    # ---------------------------------------------------------------------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I', 'R']

        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    # ---------------------------------------------------------------------
    # h_seir: S, E, I, R (no beta)p
    # ---------------------------------------------------------------------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S', 'E', 'I', 'R']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        return np.array([S, E, I, R])

    # ---------------------------------------------------------------------
    # h_incidence: incidence only (new infections per unit time)
    #             = beta (1 - u1) S I / N
    # ---------------------------------------------------------------------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['incidence']

        S    = x_vec[0]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]

        incidence = beta * (1 - u1) * S * I / self.N
        return np.array([incidence])
    
    # ---------------------------------------------------------------------
    # h_incidence_recovery: [incidence, recovery_flow]
    #   incidence = beta (1 - u1) S I / N
    #   recovery  = (gamma) I + u2 R
    # ---------------------------------------------------------------------
    def h_ir_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I', 'R', 'incidence']

        I = x_vec[2]
        R = x_vec[3]
        S    = x_vec[0]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]

        incidence = beta * (1 - u1) * S * I / self.N
        return np.array([I, R, incidence])


    # ---------------------------------------------------------------------
    # h_incidence_recovery: [incidence, recovery_flow]
    #   incidence = beta (1 - u1) S I / N
    #   recovery  = (gamma) I + u2 R
    # ---------------------------------------------------------------------
    def h_incidence_recovery(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['incidence', 'recovery']

        S    = x_vec[0]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]
        u2   = u_vec[1]

        incidence = beta * (1 - u1) * S * I / self.N
        recovery  = (self.gamma) * I + u2 * S

        return np.array([incidence, recovery])

###############################################################################
# SEIR simulation with MPC (pybounds)
###############################################################################
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
    f : callable
        Dynamics function: f(x_vec, u_vec, return_state_names=False)
    h : callable
        Measurement function: h(x_vec, u_vec, return_measurement_names=False)

    This matches usage in the Colab notebook:
        f = seiru1u3_simulation.F().f
        h = seiru1u3_simulation.H('h_ir').h
        simulate_seir(f, h, ...)
    """

    # ------------------------------------------------------------------
    # State names and input names
    # ------------------------------------------------------------------
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2']

    # ------------------------------------------------------------------
    # Measurement names
    # ------------------------------------------------------------------
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    if x0 is None:
        S0    = 0.90 * N
        E0    = 0.05 * N
        I0    = 0.04 * N
        R0    = 0#N - S0 - E0 - I0
        beta0 = 0.1#0.5
        x0    = np.array([S0, E0, I0, R0, beta0])

    # ------------------------------------------------------------------
    # Build simulator
    # ------------------------------------------------------------------
    simulator = pybounds.Simulator(f,
                                   h,
                                   dt=dt,
                                   state_names=state_names,
                                   input_names=input_names,
                                   measurement_names=measurement_names,
                                   mpc_horizon=int(10 / dt))

    # Optional measurement noise
    if measurement_noise_stds is not None:
        noise_std_array = []
        for name in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(name, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------
    tsim = np.arange(0, tsim_length, step=dt)

    # ------------------------------------------------------------------
    # Setpoint
    #   IMPORTANT: we NEVER put None in here, to avoid the .squeeze() error.
    # ------------------------------------------------------------------
    if setpoint is None:
        # Default: hold each state near its initial value (can be changed later)
        setpoint = {}
        for i, name in enumerate(state_names):
            setpoint[name] = np.ones_like(tsim) * x0[i]

    # Ensure all keys in setpoint have numeric arrays
    setpoint_processed = {}
    for key in state_names:
        if key in setpoint and setpoint[key] is not None:
            arr = np.asarray(setpoint[key]).squeeze()
            if arr.ndim == 0:  # scalar -> broadcast
                arr = np.ones_like(tsim) * float(arr)
            elif arr.shape[0] != tsim.shape[0]:
                # broadcast / trim to match tsim length
                arr = np.resize(arr, tsim.shape[0])
            setpoint_processed[key] = arr
        else:
            # if missing, just hold at initial value
            idx = state_names.index(key)
            setpoint_processed[key] = np.ones_like(tsim) * x0[idx]

    simulator.update_dict(setpoint_processed, name='setpoint')

    # ------------------------------------------------------------------
    # Cost function: penalize E and I deviations from setpoint
    # ------------------------------------------------------------------
    cost = 0
    if 'E' in state_names:
        cost += 10 * (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    if 'I' in state_names:
        cost += 100 * (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Control penalties
    simulator.mpc.set_rterm(u1=rterm_u1, u3=rterm_u3)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    eps = 1e-6

    if 'S' in state_names:
        simulator.mpc.bounds['lower', '_x', 'S'] = eps
        simulator.mpc.bounds['upper', '_x', 'S'] = N
    if 'E' in state_names:
        simulator.mpc.bounds['lower', '_x', 'E'] = eps
        simulator.mpc.bounds['upper', '_x', 'E'] = N
    if 'I' in state_names:
        simulator.mpc.bounds['lower', '_x', 'I'] = eps
        simulator.mpc.bounds['upper', '_x', 'I'] = N
    if 'R' in state_names:
        simulator.mpc.bounds['lower', '_x', 'R'] = eps
        simulator.mpc.bounds['upper', '_x', 'R'] = N
    if 'beta' in state_names:
        simulator.mpc.bounds['lower', '_x', 'beta'] = 0.1
        simulator.mpc.bounds['upper', '_x', 'beta'] = 2.0

    # Controls
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.5

    # ------------------------------------------------------------------
    # Simulate with MPC
    # ------------------------------------------------------------------
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0,
                                                    u=None,
                                                    mpc=True,
                                                    return_full_output=True)

    return t_sim, x_sim, u_sim, y_sim, simulator

###############################################################################
# Quick self-test (optional)
###############################################################################
if __name__ == "__main__":
    # Dynamics and measurement
    f = F().f
    h = H('h_ir').h

    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(f, h, tsim_length=200, dt=1.0)

    # simple plot: infectious over time
    plt.figure()
    plt.plot(t_sim, x_sim['I'])
    plt.xlabel("time (days)")
    plt.ylabel("Infectious I(t)")
    plt.title("SEIR with MPC (test run)")
    plt.tight_layout()
    plt.show()
