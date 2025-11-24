import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate

############################################################################################
# Global parameters
############################################################################################
mu = 0.02 / 365      
sigma = 1.0 / 5.2     
gamma = 1.0 / 10.0    
N = 10_000_000       

# Seasonal β parameters
beta0_default = 0.3    
epsilon_default = 0.01  
T_default = 365        

############################################################################################
# SEIR Dynamics
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
        u1 = u_vec[0]
        u3 = u_vec[1]
        t  = u_vec[2]    

        seasonal = 1.0 + self.epsilon * np.cos(2*np.pi*t/self.T)
        beta_eff = self.beta0 * seasonal * (1 - u1)

        lam = beta_eff * S * I / self.N

        dS = self.mu*self.N - lam - self.mu*S
        dE = lam - self.sigma*E - self.mu*E
        dI = self.sigma*E - (self.gamma + u3)*I - self.mu*I
        dR = (self.gamma + u3)*I - self.mu*R
        dB = 0.0   # constant

        return np.array([dS, dE, dI, dR, dB])


############################################################################################
# SEIR Measurement Functions
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
        return func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def _beta_eff_from_u(self, u_vec):
        u1 = u_vec[0]
        t  = u_vec[2]
        seasonal = 1 + self.epsilon * np.cos(2*np.pi*t/self.T)
        return self.beta0 * seasonal * (1 - u1)

    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_reported']
        return np.array([x_vec[2]])

    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names: return ['I_reported', 'new_cases']
        S, E, I, R, b = x_vec
        beta_eff = self._beta_eff_from_u(u_vec)
        return np.array([I, beta_eff*S*I/self.N])

    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured','E_measured','I_measured','R_measured']
        return np.array(x_vec[:4])

    def h_ir_newcases(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured','R_measured','new_infections']

        S,E,I,R,b = x_vec
        beta_eff = self._beta_eff_from_u(u_vec)
        return np.array([I, R, beta_eff*S*I/self.N])

    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured','E_measured','I_measured','R_measured','beta_dummy']
        return np.array(x_vec)

    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['S_measured','E_measured','I_measured','R_measured',
                    'new_infections','progressions','recoveries']

        S,E,I,R,b = x_vec
        u1, u3, t = u_vec
        beta_eff = self._beta_eff_from_u(u_vec)

        new_inf = beta_eff*S*I/self.N
        prog    = self.sigma*E
        rec     = (self.gamma+u3)*I

        return np.array([S,E,I,R,new_inf,prog,rec])


############################################################################################
# Simple Simulator (without pybounds dependency)
############################################################################################
def simulate_seir_simple(f, h, tsim_length=365, dt=1.0, x0=None):
    """
    Simple SEIR simulation without MPC control.
    Uses constant control inputs (no intervention).
    """
    
    if x0 is None:
        S0 = 0.90*N
        E0 = 0.01*N
        I0 = 0.01*N
        R0 = N - S0 - E0 - I0
        x0 = np.array([S0, E0, I0, R0, beta0_default])

    tsim = np.arange(0, tsim_length, dt)
    n_steps = len(tsim)
    
    # Initialize arrays
    x_sim = np.zeros((n_steps, len(x0)))
    u_sim = np.zeros((n_steps, 3))
    y_sim = []
    
    x_sim[0] = x0
    
    # No control (u1=0, u3=0)
    for i in range(n_steps):
        u_vec = np.array([0.0, 0.0, tsim[i]])  # [u1, u3, time]
        u_sim[i] = u_vec
        
        # Measurement
        y = h(x_sim[i], u_vec, return_measurement_names=False)
        y_sim.append(y)
        
        # State update (Euler integration)
        if i < n_steps - 1:
            dx = f(x_sim[i], u_vec, return_state_names=False)
            x_sim[i+1] = x_sim[i] + dx * dt
            
            # Ensure non-negative states
            x_sim[i+1] = np.maximum(x_sim[i+1], 0)
    
    y_sim = np.array(y_sim)
    
    return tsim, x_sim, u_sim, y_sim


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Create model instances
    f_model = F()
    h_model = H(measurement_option='h_seir')
    
    # Run simulation
    t_sim, x_sim, u_sim, y_sim = simulate_seir_simple(
        f_model, h_model, tsim_length=365, dt=1.0
    )
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot SEIR compartments
    axes[0, 0].plot(t_sim, x_sim[:, 0], label='S')
    axes[0, 0].plot(t_sim, x_sim[:, 1], label='E')
    axes[0, 0].plot(t_sim, x_sim[:, 2], label='I')
    axes[0, 0].plot(t_sim, x_sim[:, 3], label='R')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_title('SEIR Compartments')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot infected (zoomed)
    axes[0, 1].plot(t_sim, x_sim[:, 2], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Infected Population')
    axes[0, 1].set_title('Infected Individuals')
    axes[0, 1].grid(True)
    
    # Plot exposed (zoomed)
    axes[1, 0].plot(t_sim, x_sim[:, 1], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Exposed Population')
    axes[1, 0].set_title('Exposed Individuals')
    axes[1, 0].grid(True)
    
    # Plot proportions
    axes[1, 1].plot(t_sim, x_sim[:, 0]/N, label='S/N')
    axes[1, 1].plot(t_sim, x_sim[:, 1]/N, label='E/N')
    axes[1, 1].plot(t_sim, x_sim[:, 2]/N, label='I/N')
    axes[1, 1].plot(t_sim, x_sim[:, 3]/N, label='R/N')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title('Population Proportions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Simulation completed for {len(t_sim)} time steps")
    print(f"Final state: S={x_sim[-1,0]:.0f}, E={x_sim[-1,1]:.0f}, I={x_sim[-1,2]:.0f}, R={x_sim[-1,3]:.0f}")
