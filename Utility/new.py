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

# PARAMETERS FOR BETA DYNAMICS (now a state, not seasonal parameter)
beta0_default = 0.3        # baseline transmission rate
beta_decay_rate = 0.001    # decay rate for beta
beta_noise_std = 0.01      # noise/std for beta dynamics

############################################################################################
# Continuous time dynamics function
############################################################################################
class F(object):
    def __init__(self, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, beta_decay=beta_decay_rate, beta_noise=beta_noise_std):
        """Initialize with parameters stored as instance variables"""
        self.mu = mu
        self.gamma = gamma
        self.N = N

        # Beta dynamics parameters
        self.beta0 = beta0          # baseline value for initialization
        self.beta_decay = beta_decay  # decay rate towards baseline
        self.beta_noise = beta_noise  # noise/std for beta dynamics

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR model with control AND beta as state.

        States:
            x = [S, E, I, R, beta, t]

        Controls:
            u1 = u_vec[0]   social distancing (transmission reduction)
            u2 = u_vec[1]   vaccination (S -> R)
            u3 = u_vec[2]   treatment (extra recovery of I)
            u4 = u_vec[3]   beta control (optional - could be used to influence beta dynamics)

        Parameters:
            sigma = 1/10.2  (fixed progression rate from E to I)
            gamma = 1/10.0  (fixed recovery rate)

        Beta dynamics (simple mean-reverting process):
            dbeta/dt = beta_decay * (beta0 - beta) + beta_noise * ξ
            where ξ ~ N(0,1) (in discrete time, we'll handle noise separately if needed)

        Infection term (force of infection):
            lambda_inf = beta * (1 - u1) * S * I / N

        ODEs:
            dS/dt = μN - lambda_inf - u2*S - μ*S
            dE/dt = lambda_inf - σ*E - μ*E
            dI/dt = σ*E - (γ + u3)*I - μ*I
            dR/dt = (γ + u3)*I + u2*S - μ*R
            dbeta/dt = beta_decay * (beta0 - beta)
            dt/dt = 1
        """

        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta', 't']

        # Extract state variables
        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]
        t = x_vec[5]

        # Use fixed sigma parameter
        sigma = sigma_default

        # Extract controls - now 4 controls with u4 potentially affecting beta
        u1 = u_vec[0]     # prevention / social distancing
        u2 = u_vec[1]     # vaccination
        u3 = u_vec[2]     # treatment
        u4 = u_vec[3]     # beta control (optional)

        # Force of infection with time-varying beta
        # beta is now a state variable, apply social distancing control
        lambda_inf = beta * (1.0 - u1) * S * I / self.N

        # SEIR equations with controls
        dS_dt = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt = lambda_inf - sigma * E - self.mu * E
        dI_dt = sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt = (self.gamma + u3) * I + u2 * S - self.mu * R
        
        # Beta dynamics - mean-reverting process
        # u4 could represent interventions that affect transmission rate directly
        # For example: mask mandates, travel restrictions that affect beta
        dbeta_dt = self.beta_decay * (self.beta0 - beta) - u4 * beta * 0.1
        
        # Time derivative
        dt_dt = 1.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt, dt_dt])


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default):
        """
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N
        self.beta0 = beta0

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def _compute_beta_eff(self, beta, u1):
        """Helper function to compute effective transmission rate"""
        beta_eff = beta * (1.0 - u1)
        return beta_eff

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

        S  = x_vec[0]
        I  = x_vec[2]
        beta = x_vec[4]
        u1 = u_vec[0]

        beta_eff = self._compute_beta_eff(beta, u1)
        new_cases = beta_eff * S * I / self.N
        return np.array([I, new_cases])

    # -------------------------------------------------------------------------
    # 5. h_ir_new: I, R, new_cases
    # -------------------------------------------------------------------------
    def h_ir_new(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, R, new_cases]^T"""
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases']

        S  = x_vec[0]
        I  = x_vec[2]
        R  = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]

        beta_eff = self._compute_beta_eff(beta, u1)
        new_cases = beta_eff * S * I / self.N
        return np.array([I, R, new_cases])

    # -------------------------------------------------------------------------
    # 6. h_ei_flows: E, I, new_inf, prog
    # -------------------------------------------------------------------------
    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [E, I, new_inf, prog]^T"""
        if return_measurement_names:
            return ['E_measured', 'I_measured', 'new_inf', 'prog']

        S  = x_vec[0]
        E  = x_vec[1]
        I  = x_vec[2]
        beta = x_vec[4]
        u1 = u_vec[0]

        beta_eff = self._compute_beta_eff(beta, u1)
        new_inf = beta_eff * S * I / self.N
        prog = sigma_default * E

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

        S  = x_vec[0]
        E  = x_vec[1]
        I  = x_vec[2]
        R  = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]

        beta_eff = self._compute_beta_eff(beta, u1)
        new_inf = beta_eff * S * I / self.N
        return np.array([S, E, I, R, new_inf])

    # -------------------------------------------------------------------------
    # 9. h_seir_with_beta: S, E, I, R, beta_eff
    # -------------------------------------------------------------------------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, beta_eff]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_eff']

        S  = x_vec[0]
        E  = x_vec[1]
        I  = x_vec[2]
        R  = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]

        beta_eff = self._compute_beta_eff(beta, u1)
        return np.array([S, E, I, R, beta_eff])

    # -------------------------------------------------------------------------
    # 10. h_with_flows: S, E, I, R, new_inf, prog, recov
    # -------------------------------------------------------------------------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, new_inf, prog, recov]^T"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf', 'prog', 'recov']

        S  = x_vec[0]
        E  = x_vec[1]
        I  = x_vec[2]
        R  = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]
        u3 = u_vec[2]

        beta_eff = self._compute_beta_eff(beta, u1)
        new_inf = beta_eff * S * I / self.N
        prog = sigma_default * E
        recov = (self.gamma + u3) * I

        return np.array([S, E, I, R, new_inf, prog, recov])

    # -------------------------------------------------------------------------
    # 11. h_observable: I, R, new_cases, recov
    # -------------------------------------------------------------------------
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [I, R, new_cases, recov]^T"""
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases', 'recov']

        S  = x_vec[0]
        I  = x_vec[2]
        R  = x_vec[3]
        beta = x_vec[4]
        u1 = u_vec[0]
        u3 = u_vec[2]
        
        beta_eff = self._compute_beta_eff(beta, u1)
        recov = (self.gamma + u3) * I
        new_cases = beta_eff * S * I / self.N

        return np.array([I, R, new_cases, recov])
    
    # -------------------------------------------------------------------------
    # 12. h_beta_direct: beta (direct measurement of transmission rate)
    # -------------------------------------------------------------------------
    def h_beta_direct(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [beta] (direct measurement of transmission rate)"""
        if return_measurement_names:
            return ['beta_measured']
        beta = x_vec[4]
        return np.array([beta])
    
    # -------------------------------------------------------------------------
    # 13. h_seir_beta: S, E, I, R, beta (all states)
    # -------------------------------------------------------------------------
    def h_seir_beta(self, x_vec, u_vec, return_measurement_names=False):
        """Measurement: y = [S, E, I, R, beta]^T (all states)"""
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_measured']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        beta = x_vec[4]

        return np.array([S, E, I, R, beta])


############################################################################################
# SEIR simulation with MPC (modified for beta as state)
############################################################################################
def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None,
                  setpoint=None, rterm_u1=1e-4, rterm_u2=1e-4, rterm_u3=1e-4, rterm_u4=1e-4,
                  x0=None, measurement_noise_stds=None):

    # Default initial conditions (now includes beta)
    if x0 is None:
        S0 = 0.70 * N
        E0 = 0.05 * N
        I0 = 0.05 * N
        R0 = N - S0 - E0 - I0
        beta0 = beta0_default  # Initial beta value
        t0 = 0.0

        x0 = np.array([S0, E0, I0, R0, beta0, t0])

    # State and input names (now 4 controls)
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3', 'u4']  # Added u4 for beta control

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

    # Default setpoint (now includes beta)
    if setpoint is None:
        I_initial = x0[2]
        E_initial = x0[1]
        beta_initial = x0[4]

        I_target = 0.0001 * N
        E_target = 0.00005 * N
        beta_target = 0.1  # Target for beta (lower transmission)

        I_set = I_target + (I_initial - I_target) * np.exp(-tsim / 100.0)
        E_set = E_target + (E_initial - E_target) * np.exp(-tsim / 80.0)
        beta_set = beta_target + (beta_initial - beta_target) * np.exp(-tsim / 150.0)

        setpoint = {
            'S': np.zeros_like(tsim),
            'E': E_set,
            'I': I_set,
            'R': np.zeros_like(tsim),
            'beta': beta_set,
            't': tsim  # Time state tracks simulation time
        }

    simulator.update_dict(setpoint, name='setpoint')

    # MPC cost (penalize E, I, and beta)
    cost_E = (simulator.model.x['E'] - simulator.model.tvp['E_set'])**2
    cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set'])**2
    cost_beta = (simulator.model.x['beta'] - simulator.model.tvp['beta_set'])**2
    cost = 10.0 * cost_E + 100.0 * cost_I + 1.0 * cost_beta

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Input penalties (now 4 controls)
    simulator.mpc.set_rterm(u1=rterm_u1, u2=rterm_u2, u3=rterm_u3, u4=rterm_u4)

    # State bounds (now includes beta)
    eps = 1e-6
    simulator.mpc.bounds['lower', '_x', 'S'] = eps
    simulator.mpc.bounds['upper', '_x', 'S'] = N
    simulator.mpc.bounds['lower', '_x', 'E'] = eps
    simulator.mpc.bounds['upper', '_x', 'E'] = N
    simulator.mpc.bounds['lower', '_x', 'I'] = eps
    simulator.mpc.bounds['upper', '_x', 'I'] = N
    simulator.mpc.bounds['lower', '_x', 'R'] = eps
    simulator.mpc.bounds['upper', '_x', 'R'] = N
    simulator.mpc.bounds['lower', '_x', 'beta'] = 0.0  # Beta cannot be negative
    simulator.mpc.bounds['upper', '_x', 'beta'] = 1.0  # Upper bound for beta
    
    # Bounds for time state
    simulator.mpc.bounds['lower', '_x', 't'] = 0.0
    simulator.mpc.bounds['upper', '_x', 't'] = tsim_length + 100.0

    # Control bounds (now 4 controls)
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 0.9
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.6
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.5
    simulator.mpc.bounds['lower', '_u', 'u4'] = 0.0  # Beta control (e.g., 0 = no direct intervention)
    simulator.mpc.bounds['upper', '_u', 'u4'] = 0.5  # Max intervention on beta

    # Run simulation with MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


# Example usage with beta as state
def example_simulation():
    # Create dynamics and measurement objects
    f_obj = F(beta0=0.3, beta_decay=0.001, beta_noise=0.01)
    h_obj = H('h_seir_beta')  # Measure all states including beta
    
    # Run simulation
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seir(
        f_obj, h_obj, 
        tsim_length=200,
        dt=1.0,
        measurement_names=['S_measured', 'E_measured', 'I_measured', 'R_measured', 'beta_measured']
    )
    
    # Extract states
    S = x_sim[:, 0]
    E = x_sim[:, 1]
    I = x_sim[:, 2]
    R = x_sim[:, 3]
    beta = x_sim[:, 4]
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot SEIR compartments
    axes[0, 0].plot(t_sim, S/N, label='S')
    axes[0, 0].plot(t_sim, E/N, label='E')
    axes[0, 0].plot(t_sim, I/N, label='I')
    axes[0, 0].plot(t_sim, R/N, label='R')
    axes[0, 0].set_title('SEIR Compartments')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Fraction of Population')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot beta (transmission rate)
    axes[0, 1].plot(t_sim, beta, label='β(t)', color='red')
    axes[0, 1].axhline(y=beta0_default, color='gray', linestyle='--', label='β₀ baseline')
    axes[0, 1].set_title('Transmission Rate β(t)')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('β')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot controls
    axes[1, 0].plot(t_sim[:-1], u_sim[:, 0], label='u1 (social distancing)')
    axes[1, 0].plot(t_sim[:-1], u_sim[:, 1], label='u2 (vaccination)')
    axes[1, 0].plot(t_sim[:-1], u_sim[:, 2], label='u3 (treatment)')
    axes[1, 0].plot(t_sim[:-1], u_sim[:, 3], label='u4 (beta control)')
    axes[1, 0].set_title('Control Inputs')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Control Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot new infections
    beta_eff = beta[:-1] * (1 - u_sim[:, 0])
    new_infections = beta_eff * S[:-1] * I[:-1] / N
    axes[1, 1].plot(t_sim[:-1], new_infections, label='New infections/day')
    axes[1, 1].set_title('New Infections per Day')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Number of People')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot effective reproduction number
    effective_R0 = beta_eff / (gamma + mu)
    axes[2, 0].plot(t_sim[:-1], effective_R0, label='Effective R₀', color='purple')
    axes[2, 0].axhline(y=1.0, color='red', linestyle='--', label='R₀ = 1 threshold')
    axes[2, 0].set_title('Effective Reproduction Number')
    axes[2, 0].set_xlabel('Time (days)')
    axes[2, 0].set_ylabel('R₀')
    axes[2, 0].legend()
    axes[2, 1].grid(True)
    
    # Plot total infected over time
    total_infected = E + I
    axes[2, 1].plot(t_sim, total_infected/N, label='E+I (total infected)', color='orange')
    axes[2, 1].set_title('Total Infected (E+I)')
    axes[2, 1].set_xlabel('Time (days)')
    axes[2, 1].set_ylabel('Fraction of Population')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return t_sim, x_sim, u_sim, y_sim, simulator

# Run the example
if __name__ == "__main__":
    example_simulation()
