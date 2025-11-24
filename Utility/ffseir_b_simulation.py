# ===============================================================
# FULL SEASONAL-B SEIR MODEL WITH MPC (COMPATIBLE WITH YOUR NEW
# simulate_seir FUNCTION)
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

beta0_default = 0.3
epsilon_default = 0.01
T_default = 365



###############################################################
# 1. SEIR DYNAMICS CLASS — STATE VECTOR = [S, E, I, R, beta]
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

        # seasonal forcing parameters
        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def __call__(self, x_vec, u_vec, return_state_names=False):
        return self.f(x_vec, u_vec, return_state_names)

    def f(self, x_vec, u_vec, return_state_names=False):
        """Dynamics f(x,u) fitting the NEW simulate_seir() setup."""

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # unpack state
        S, E, I, R, beta_state = x_vec
        u1, u2 = u_vec    # vaccination + treatment

        # Seasonal transmission multiplier
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * beta_state / self.T)

        # effective beta
        beta_eff = self.beta0 * seasonal * (1 - u1)

        # force of infection
        lam = beta_eff * S * I / self.N

        # ODE system
        dS = self.mu*self.N - lam - self.mu*S - u2*S
        dE = lam - self.sigma*E - self.mu*E
        dI = self.sigma*E - self.gamma*I - self.mu*I
        dR = self.gamma*I + u2*S - self.mu*R

        # β-state is constant (or later can be dynamic)
        dBeta = 0.0

        return np.array([dS, dE, dI, dR, dBeta])



###############################################################
# 2. MEASUREMENT CLASS (H)
###############################################################
class H(object):
    def __init__(self, measurement_option, N=N):
        self.measurement_option = measurement_option
        self.N = N

    def __call__(self, x_vec, u_vec, return_measurement_names=False):
        return self.h(x_vec, u_vec, return_measurement_names)

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = getattr(self, self.measurement_option)
        return func(x_vec, u_vec, return_measurement_names)

    # Example measurement: I and R
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        S,E,I,R,beta = x_vec
        return np.array([I, R])



###############################################################
# 3. SIMULATOR (YOUR NEW MPC-SAFE VERSION)
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
    Fully compatible with the new structure you posted.
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
    # Initial Conditions
    # -----------------------------------------------------------
    if x0 is None:
        S0 = 0.90 * N
        E0 = 0.05 * N
        I0 = 0.04 * N
        R0 = 0
        beta0 = 0.1
        x0 = np.array([S0, E0, I0, R0, beta0])


    # -----------------------------------------------------------
    # Build pybounds Simulator
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
    # Setpoint handling (your safe logic)
    # -----------------------------------------------------------
    if setpoint is None:
        setpoint = {name: np.ones_like(tsim)*x0[i]
                    for i, name in enumerate(state_names)}

    setpoint_processed = {}
    for key in state_names:
        arr = np.asarray(setpoint[key]).squeeze()
        if arr.ndim == 0:
            arr = np.ones_like(tsim)*float(arr)
        else:
            arr = np.resize(arr, tsim.shape[0])
        setpoint_processed[key] = arr

    simulator.update_dict(setpoint_processed, name='setpoint')


    # -----------------------------------------------------------
    # COST FUNCTION (E and I tracking)
    # -----------------------------------------------------------
    cost = 0
    if 'E' in state_names:
        cost += 10 * (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    if 'I' in state_names:
        cost += 100 * (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2)


    # -----------------------------------------------------------
    # STATE BOUNDS
    # -----------------------------------------------------------
    eps = 1e-6
    for var in state_names:
        if var == 'beta':
            simulator.mpc.bounds['lower','_x','beta'] = 0.1
            simulator.mpc.bounds['upper','_x','beta'] = 2.0
        else:
            simulator.mpc.bounds['lower','_x',var] = eps
            simulator.mpc.bounds['upper','_x',var] = N


    # -----------------------------------------------------------
    # CONTROL BOUNDS
    # -----------------------------------------------------------
    simulator.mpc.bounds['lower','_u','u1'] = 0.0
    simulator.mpc.bounds['upper','_u','u1'] = 0.9

    simulator.mpc.bounds['lower','_u','u2'] = 0.0
    simulator.mpc.bounds['upper','_u','u2'] = 0.5


    # -----------------------------------------------------------
    # SIMULATION
    # -----------------------------------------------------------
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator



###############################################################
# 4. QUICK TEST RUN
###############################################################
if __name__ == "__main__":
    f = F()
    h = H('h_ir')

    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
        f,
        h,
        tsim_length=200,
        dt=1.0
    )

    plt.plot(t_sim, x_sim['I'])
    plt.xlabel("Time (days)")
    plt.ylabel("Infectious I(t)")
    plt.title("SEIR Seasonal-beta with MPC")
    plt.grid(True)
    plt.show()
