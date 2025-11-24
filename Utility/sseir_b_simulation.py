# ===============================================================
# SEASONAL-BETA SEIR MODEL WITH MULTIPLE MEASUREMENTS + MPC
# Fixed version for observability analysis
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

beta0_default = 0.5
epsilon_default = 0.2
T_default = 365


###############################################################
# 1. SEIR DYNAMICS f(x, u)
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

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_dummy']

        S, E, I, R, beta_dummy = x_vec
        u1, u3, t = u_vec

        # Seasonal transmission
        seasonal = 1 + self.epsilon * np.cos(2 * np.pi * t / self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)

        lambda_inf = beta_eff * S * I / self.N

        dS = self.mu * self.N - lambda_inf - self.mu * S
        dE = lambda_inf - self.sigma * E - self.mu * E
        dI = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR = (self.gamma + u3) * I - self.mu * R

        dBeta = 0.0
        return np.array([dS, dE, dI, dR, dBeta])


###############################################################
# 2. MEASUREMENT FUNCTIONS H(x, u)
###############################################################
class H(object):

    def __init__(self, measurement_option,
                 mu=mu, sigma=sigma, gamma=gamma, N=N,
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
    def _beta_eff(self, x_vec, u_vec):
        S, E, I, R, b = x_vec
        u1, u3, t = u_vec
        seasonal = 1 + self.epsilon * np.cos(2*np.pi*t/self.T)
        return self.beta0 * seasonal * (1 - u1)

    # ---------------- MEASUREMENTS ------------------

    def h_ir_newcases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_infections']
        S, E, I, R, b = x_vec
        beta_eff = self._beta_eff(x_vec, u_vec)
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
        beta_eff = self._beta_eff(x_vec, u_vec)
        return np.array([I, beta_eff*S*I/self.N])

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S', 'E', 'I', 'R']
        return np.array(x_vec[:4])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S', 'E', 'I', 'R', 'beta_dummy']
        return np.array(x_vec)

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S', 'E', 'I', 'R', 'new_infections', 'progressions', 'recoveries']
        S, E, I, R, b = x_vec
        beta_eff = self._beta_eff(x_vec, u_vec)
        new_inf = beta_eff*S*I/self.N
        prog = self.sigma*E
        rec = (self.gamma+u_vec[1])*I
        return np.array([S, E, I, R, new_inf, prog, rec])


###############################################################
# 3. MPC SIMULATOR (FIXED FOR PYBOUNDS)
###############################################################
def simulate_seir(f, h,
                  tsim_length=365, dt=1.0,
                  measurement_names=None,
                  setpoint=None,
                  rterm_u1=1e-4,
                  rterm_u3=1e-4,
                  x0=None,
                  measurement_noise_stds=None):

    # initial conditions
    if x0 is None:
        S0 = 0.90*N
        E0 = 0.01*N
        I0 = 0.01*N
        R0 = N - S0 - E0 - I0
        beta0 = beta0_default
        x0 = np.array([S0, E0, I0, R0, beta0])

    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u3', 'time']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10/dt)
    )

    if measurement_noise_stds is not None:
        simulator.measurement_noise_std = np.array([
            measurement_noise_stds.get(name, 0.0)
            for name in measurement_names
        ])

    tsim = np.arange(0, tsim_length, dt)

    # ================================================================
    # CRITICAL FIX: Only create setpoints for E and I
    # These are the ONLY states referenced in the cost function below
    # ================================================================
    if setpoint is None:
        setpoint = {}
        if 'E' in state_names:
            idx_E = state_names.index('E')
            setpoint['E'] = np.ones_like(tsim) * x0[idx_E]
        if 'I' in state_names:
            idx_I = state_names.index('I')
            setpoint['I'] = np.ones_like(tsim) * x0[idx_I]
    
    # Process setpoint - ONLY for E and I (not S, R, or beta_dummy)
    setpoint_processed = {}
    for key in ['E', 'I']:  # ONLY these two states
        if key in setpoint and setpoint[key] is not None:
            arr = np.asarray(setpoint[key]).squeeze()
            if arr.ndim == 0:  # scalar -> broadcast
                arr = np.ones_like(tsim) * float(arr)
            elif arr.shape[0] != tsim.shape[0]:
                arr = np.resize(arr, tsim.shape[0])
            setpoint_processed[key] = arr
        elif key in state_names:
            idx = state_names.index(key)
            setpoint_processed[key] = np.ones_like(tsim) * x0[idx]

    simulator.update_dict(setpoint_processed, name='setpoint')

    # Cost function - only references E and I
    cost = 10*(simulator.model.x['E'] - simulator.model.tvp['E_set'])**2 \
         + 100*(simulator.model.x['I'] - simulator.model.tvp['I_set'])**2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=rterm_u1, u3=rterm_u3)

    # bounds
    eps = 1e-6
    for v in ['S', 'E', 'I', 'R']:
        simulator.mpc.bounds['lower', '_x', v] = eps
        simulator.mpc.bounds['upper', '_x', v] = N
    simulator.mpc.bounds['lower', '_x', 'beta_dummy'] = 0.1
    simulator.mpc.bounds['upper', '_x', 'beta_dummy'] = 2.0

    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.5

    simulator.mpc.bounds['lower', '_u', 'time'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'time'] = tsim_length

    return simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )


###############################################################################
# Quick self-test
###############################################################################
if __name__ == "__main__":
    print("Testing sseir_b_simulation module...")
    
    f = F().f
    h = H('h_ir_newcases').h
    
    print("Available measurements:", h(None, None, return_measurement_names=True))
    
    measurement_noise_stds = {
        'I_measured': 6000,
        'R_measured': 5000,
        'new_infections': 4000
    }
    
    print("Running simulation...")
    t_sim, x_sim, u_sim, y_sim = simulate_seir(
        f,
        h,
        tsim_length=365,
        dt=1.0,
        measurement_noise_stds=measurement_noise_stds
    )
    
    print(f"Simulation completed successfully!")
    print(f"Time steps: {len(t_sim)}")
    print(f"States: {list(x_sim.keys())}")
    print(f"Controls: {list(u_sim.keys())}")
    print(f"Measurements: {list(y_sim.keys())}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(t_sim, x_sim['I'])
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious')
    axes[0, 0].set_title('Infectious Population')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t_sim, x_sim['R'])
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Recovered')
    axes[0, 1].set_title('Recovered Population')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(t_sim, u_sim['u1'], label='u1 (prevention)')
    axes[1, 0].plot(t_sim, u_sim['u3'], label='u3 (treatment)')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Control')
    axes[1, 0].set_title('Control Inputs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t_sim, y_sim['new_infections'])
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('New Infections/day')
    axes[1, 1].set_title('Daily New Infections')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
