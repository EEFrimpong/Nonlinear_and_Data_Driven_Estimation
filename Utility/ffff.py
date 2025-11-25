import numpy as np
import matplotlib.pyplot as plt
import pybounds

############################################################################################
# Global parameters
############################################################################################
mu = 0.02 / 365            # Natural mortality rate per day (2% per year)
sigma_default = 1.0 / 5.2  # Default progression rate from E to I (5.2 days)
gamma = 1.0 / 10.0         # Recovery rate (10 days infectious period)
N = 1_000_000              # Total population

# Seasonal structure parameters for beta_eff
beta0_default = 0.5        # Baseline transmission
epsilon_default = 0.2      # Seasonal amplitude
T_default = 365            # Seasonal period (days)


############################################################################################
# Continuous-time dynamics function F
############################################################################################
class F(object):
    def __init__(self, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):
        """
        States:
            x = [S, E, I, R, beta_eff, sigma, t]

        Controls:
            u1 : social distancing (transmission reduction)
            u2 : vaccination (S -> R)
            u3 : treatment (extra recovery of I)

        Dynamics:
            dS/dt          = μN - β_eff (1 - u1) S I / N - u2 S - μS
            dE/dt          = β_eff (1 - u1) S I / N - σ E - μE
            dI/dt          = σ E - (γ + u3) I - μI
            dR/dt          = (γ + u3) I + u2 S - μR
            dβ_eff/dt      = β0 * (-ε * 2π/T) sin(2π t/T) (1 - u1)
            dσ/dt          = 0
            dt/dt          = 1
        """
        self.mu = mu
        self.gamma = gamma
        self.N = N
        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def f(self, x_vec, u_vec, return_state_names=False):
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta_eff', 'sigma', 't']

        # unpack states
        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        t        = x_vec[6]

        # controls
        u1 = u_vec[0]   # distancing
        u2 = u_vec[1]   # vaccination
        u3 = u_vec[2]   # treatment

        # force of infection
        lambda_inf = beta_eff * (1.0 - u1) * S * I / self.N

        # SEIR with controls
        dS_dt = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - sigma * E - self.mu * E
        dI_dt = sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R

        # seasonal beta_eff evolution
        dbeta_dt = (
            self.beta0 *
            (-self.epsilon) * (2.0 * np.pi / self.T) *
            np.sin(2.0 * np.pi * t / self.T) *
            (1.0 - u1)
        )

        # sigma state (constant here)
        dsigma_dt = 0.0

        # time state
        dt_dt = 1.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt, dsigma_dt, dt_dt])


############################################################################################
# Measurement functions H
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N):
        """
        measurement_option: name of measurement function:
            'h_reported_cases'
            'h_incidence'
            'h_ir'
            'h_seir_with_beta'
            'h_with_flows'
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = getattr(self, self.measurement_option)
        return func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """ y = [I] """
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """ y = [I, new_cases] """
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S        = x_vec[0]
        I        = x_vec[2]
        beta_eff = x_vec[4]
        u1       = u_vec[0]

        new_cases = beta_eff * (1.0 - u1) * S * I / self.N
        return np.array([I, new_cases])

    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """ y = [I, R] """
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """ y = [S, E, I, R, beta_eff, sigma] """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured',
                    'R_measured', 'beta_eff', 'sigma']
        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        return np.array([S, E, I, R, beta_eff, sigma])

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        y = [I, R, new_infections, progressions, recoveries]

        new_infections = beta_eff (1 - u1) S I / N
        progressions   = sigma E
        recoveries     = (gamma + u3) I
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured',
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

        return np.array([I, R, new_inf, progress, recoveries])


############################################################################################
# SEIR simulation with MPC (pybounds)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None,
                  rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4,
                  x0=None, measurement_noise_stds=None):
    """
    f : callable
        Dynamics function: f(x_vec, u_vec, return_state_names=False)
    h : callable
        Measurement function: h(x_vec, u_vec, return_measurement_names=False)

    Returns:
        t_sim, x_sim, u_sim, y_sim, simulator
    """

    # ------------------- initial conditions -------------------
    if x0 is None:
        S0 = 0.90 * N
        E0 = 0.01 * N
        I0 = 0.01 * N
        R0 = N - S0 - E0 - I0
        t0 = 0.0

        beta_eff0 = beta0_default * (1.0 + epsilon_default * np.cos(0.0))
        sigma0    = sigma_default

        x0 = np.array([S0, E0, I0, R0, beta_eff0, sigma0, t0])

    # ------------------- names -------------------
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # ------------------- build simulator -------------------
    simulator = pybounds.Simulator(
        f, h,
        dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10 / dt)
    )

    # ------------------- measurement noise -------------------
    if measurement_noise_stds is not None:
        noise_std_array = []
        for name in measurement_names:
            noise_std_array.append(measurement_noise_stds.get(name, 0.0))
        simulator.measurement_noise_std = np.array(noise_std_array)

    # ------------------- time grid -------------------
    tsim = np.arange(0, tsim_length, step=dt)

    # ------------------- setpoint processing -------------------
    # IMPORTANT: we NEVER put None in here, to avoid the .squeeze() error.
    if setpoint is None:
        # Default: hold each state near its initial value
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

    # ------------------- cost function -------------------
    # Penalize E and I deviations from setpoint
    cost = 0
    if 'E' in state_names:
        cost += 10 * (simulator.model.x['E'] - simulator.model.tvp['E_set']) ** 2
    if 'I' in state_names:
        cost += 100 * (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # input regularization
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3)

    # ------------------- bounds -------------------
    eps = 1e-6

    for var in ['S', 'E', 'I', 'R']:
        if var in state_names:
            simulator.mpc.bounds['lower', '_x', var] = eps
            simulator.mpc.bounds['upper', '_x', var] = N

    if 'beta_eff' in state_names:
        simulator.mpc.bounds['lower', '_x', 'beta_eff'] = 0.1
        simulator.mpc.bounds['upper', '_x', 'beta_eff'] = 2.0
    if 'sigma' in state_names:
        simulator.mpc.bounds['lower', '_x', 'sigma'] = 1.0 / 30.0
        simulator.mpc.bounds['upper', '_x', 'sigma'] = 1.0
    if 't' in state_names:
        simulator.mpc.bounds['lower', '_x', 't'] = 0.0
        simulator.mpc.bounds['upper', '_x', 't'] = tsim_length + 10.0

    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.6
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.5

    # ------------------- simulate -------------------
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Create dynamics and measurement functions
    f = F().f
    h_e = H('h_ir').h
    
    # Define measurement noise
    measurement_noise_stds = {
        'I_measured': 100,
        'R_measured': 500
    }
    
    # Run simulation
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
        f,
        h_e,
        tsim_length=365,
        dt=1.0,
        measurement_noise_stds=measurement_noise_stds
    )
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot SEIR states
    axes[0].plot(t_sim, x_sim[:, 0], label='S', alpha=0.7)
    axes[0].plot(t_sim, x_sim[:, 1], label='E', alpha=0.7)
    axes[0].plot(t_sim, x_sim[:, 2], label='I', alpha=0.7)
    axes[0].plot(t_sim, x_sim[:, 3], label='R', alpha=0.7)
    axes[0].set_ylabel('Population')
    axes[0].set_title('SEIR Compartments')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot controls
    axes[1].plot(t_sim, u_sim[:, 0], label='u1 (distancing)', alpha=0.7)
    axes[1].plot(t_sim, u_sim[:, 1], label='u2 (vaccination)', alpha=0.7)
    axes[1].plot(t_sim, u_sim[:, 2], label='u3 (treatment)', alpha=0.7)
    axes[1].set_ylabel('Control values')
    axes[1].set_title('Control Interventions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot beta_eff and sigma
    axes[2].plot(t_sim, x_sim[:, 4], label='beta_eff', alpha=0.7)
    axes[2].plot(t_sim, x_sim[:, 5], label='sigma', alpha=0.7)
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Parameter values')
    axes[2].set_title('Time-varying Parameters')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
