import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize

from scipy import interpolate

import pandas as pd

import pybounds

############################################################################################
# Set some global parameters for SEIR model with Vaccination
############################################################################################
F = 10.0      # recruitment/birth rate (individuals per time unit)
beta = 0.0005  # transmission rate 
mu = 0.02      # natural mortality rate
c = 0.2        # progression rate from E to I
r = 0.05       # recovery rate

############################################################################################
# Continuous time dynamics function - SEIR-V model with 3 controls
############################################################################################
class F(object):
    def __init__(self, beta=beta, mu=mu, c=c, r=r, F_rate=F):
        """
        Initialize SEIR-V model parameters
        
        Parameters:
        beta : transmission rate
        mu : natural mortality rate  
        c : progression rate E->I
        r : recovery rate
        F_rate : recruitment/birth rate
        """
        self.beta = beta
        self.mu = mu
        self.c = c
        self.r = r
        self.F_rate = F_rate

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        Continuous time dynamics function for SEIR-V epidemic model with 3 controls.
        
        State dynamics:
        Ṡ = F - β(1-u₁)SI - u₂S - μS
        Ė = β(1-u₁)SI - (μ+c)E
        İ = cE - (μ+r+u₃)I
        Ṙ = (r+u₃)I - μR
        V̇ = u₂S - μV
        
        This is written in control-affine form: 
        ẋ = f₀(x) + f₁(x)u₁ + f₂(x)u₂ + f₃(x)u₃
        
        Parameters:
        x_vec : array-like, shape (5,)
            State vector [S, E, I, R, V]
        u_vec : array-like, shape (3,)
            Control vector [u1, u2, u3]
            u1: transmission reduction effort (0 to 1)
            u2: vaccination rate (≥ 0)
            u3: treatment rate (≥ 0)
        
        Returns:
        x_dot : numpy array, shape (5,)
            Time derivative of state vector
        """
        
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'V']
        
        # Extract state variables
        S = x_vec[0]  # Susceptible
        E = x_vec[1]  # Exposed
        I = x_vec[2]  # Infected
        R = x_vec[3]  # Recovered
        V = x_vec[4]  # Vaccinated

        # Extract control inputs
        u1 = u_vec[0]  # Transmission reduction (0-1)
        u2 = u_vec[1]  # Vaccination rate
        u3 = u_vec[2]  # Treatment rate
        
        # f0 component: drift dynamics (no controls, i.e., u1=0, u2=0, u3=0)
        f0_contribution = np.array([
            self.F_rate - self.beta * S * I - self.mu * S,     # Ṡ
            self.beta * S * I - (self.mu + self.c) * E,        # Ė
            self.c * E - (self.mu + self.r) * I,               # İ (no treatment)
            self.r * I - self.mu * R,                          # Ṙ (no treatment)
            -self.mu * V                                        # V̇ (no vaccination)
        ])
        
        # f1 component: multiplied by control u1 (transmission reduction)
        # u1 reduces transmission: -β(1-u1)SI = -βSI + βu1·SI
        # So the u1 term contributes: +β·SI·u1
        f1_contribution = u1 * np.array([
            self.beta * S * I,      # +βSI term in Ṡ
            -self.beta * S * I,     # -βSI term in Ė
            0,                       # no u1 term in İ
            0,                       # no u1 term in Ṙ
            0                        # no u1 term in V̇
        ])
        
        # f2 component: multiplied by control u2 (vaccination)
        f2_contribution = u2 * np.array([
            -S,          # -u₂·S term in Ṡ
            0,           # no u2 term in Ė
            0,           # no u2 term in İ
            0,           # no u2 term in Ṙ
            S            # +u₂·S term in V̇
        ])

        # f3 component: multiplied by control u3 (treatment)
        f3_contribution = u3 * np.array([
            0,           # no u3 term in Ṡ
            0,           # no u3 term in Ė
            -I,          # -u₃·I term in İ (faster removal)
            I,           # +u₃·I term in Ṙ (moves to recovered)
            0            # no u3 term in V̇
        ])

        # Combined dynamics: ẋ = f₀(x) + f₁(x)u₁ + f₂(x)u₂ + f₃(x)u₃
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution + f3_contribution
        
        return x_dot_vec


############################################################################################
# Continuous time measurement functions
############################################################################################
class H(object):
    def __init__(self, measurement_option):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_all_states(self, x_vec, u_vec, return_measurement_names=False):
        """Measure all states directly"""
        if return_measurement_names:
            return ['S', 'E', 'I', 'R', 'V']

        y_vec = np.array(x_vec)  # Direct measurement of all 5 states
        return y_vec

    def h_infected_only(self, x_vec, u_vec, return_measurement_names=False):
        """Measure only infected population (most commonly observable)"""
        if return_measurement_names:
            return ['I']

        I = x_vec[2]
        y_vec = np.array([I])
        return y_vec

    def h_infected_recovered(self, x_vec, u_vec, return_measurement_names=False):
        """Measure infected and recovered (hospital data + testing)"""
        if return_measurement_names:
            return ['I', 'R']

        I = x_vec[2]
        R = x_vec[3]
        y_vec = np.array([I, R])
        return y_vec

    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        """Measure infected and recovered (alias for h_infected_recovered)"""
        return self.h_infected_recovered(x_vec, u_vec, return_measurement_names)

    def h_new_infections(self, x_vec, u_vec, return_measurement_names=False):
        """Measure rate of new infections: β(1-u1)SI"""
        if return_measurement_names:
            return ['new_infections', 'I']

        S = x_vec[0]
        I = x_vec[2]
        u1 = u_vec[0]
        
        beta = 0.0005  # transmission rate (should match model parameter)
        new_infections = beta * (1 - u1) * S * I
        
        y_vec = np.array([new_infections, I])
        return y_vec

    def h_epidemic_key_indicators(self, x_vec, u_vec, return_measurement_names=False):
        """Measure key epidemic indicators: I, R, and vaccination coverage V/N"""
        if return_measurement_names:
            return ['I', 'R', 'vaccination_coverage']

        S = x_vec[0]
        E = x_vec[1]
        I = x_vec[2]
        R = x_vec[3]
        V = x_vec[4]
        
        total_pop = S + E + I + R + V
        vaccination_coverage = V / total_pop if total_pop > 0 else 0
        
        y_vec = np.array([I, R, vaccination_coverage])
        return y_vec


############################################################################################
# SEIR-V simulation
############################################################################################
def simulate_seirv(f, h, tsim_length=200, dt=1.0, measurement_names=None,
                  control_strategy='suppress_outbreak', rterm=1e-2, 
                  measurement_noise_stds=None, setpoint=None):
    """
    Simulate SEIR-V epidemic model with MPC control
    
    control_strategy: 'suppress_outbreak', 'flatten_curve', 'minimal_intervention'
    measurement_noise_stds: dict of measurement noise standard deviations
    setpoint: dict of custom setpoints (optional)
    """
    # Set state and input names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']  # u1: transmission reduction, u2: vaccination, u3: treatment
    
    # Choose the measurement function
    if measurement_names is None:
        try:
            measurement_names = h(None, None, return_measurement_names=True) 
        except:
            raise ValueError('Need to provide measurement_names as a list of strings')

    # Initialize simulator
    simulator = pybounds.Simulator(f, h, dt=dt, state_names=state_names, 
                                   input_names=input_names, measurement_names=measurement_names, 
                                   mpc_horizon=int(20/dt))
    
    # Set measurement noise if provided
    if measurement_noise_stds is not None:
        simulator.measurement_noise_stds = measurement_noise_stds

    # Define the set-point(s) to follow
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    if setpoint is None:
        if control_strategy == 'suppress_outbreak':
            # Goal: minimize infected population
            setpoint = {
                'S': NA,
                'E': NA,
                'I': np.ones_like(tsim) * 1.0,  # Target: keep infections very low
                'R': NA,
                'V': NA,
            }
            # Cost focuses on minimizing I
            cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
            cost = 100 * cost_I  # High weight on infection control
            
        elif control_strategy == 'flatten_curve':
            # Goal: moderate infected levels to avoid overwhelming healthcare
            target_I = 50.0  # Target infection level
            setpoint = {
                'S': NA,
                'E': NA,
                'I': np.ones_like(tsim) * target_I,
                'R': NA,
                'V': NA,
            }
            cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
            cost = cost_I
            
        elif control_strategy == 'minimal_intervention':
            # Goal: let epidemic run with minimal control cost
            setpoint = {
                'S': NA,
                'E': NA,
                'I': NA,
                'R': NA,
                'V': NA,
            }
            # Small cost on states, main cost on controls
            cost_I = simulator.model.x['I'] ** 2
            cost = 0.1 * cost_I
            
        elif control_strategy == 'vaccination_focus':
            # Goal: maximize vaccination coverage while controlling outbreak
            target_V_coverage = 0.7  # 70% vaccination coverage
            total_pop = simulator.model.x['S'] + simulator.model.x['E'] + simulator.model.x['I'] + simulator.model.x['R'] + simulator.model.x['V']
            V_coverage = simulator.model.x['V'] / total_pop
            
            setpoint = {
                'S': NA,
                'E': NA,
                'I': np.ones_like(tsim) * 10.0,  # Keep infections low
                'R': NA,
                'V': NA,
            }
            cost_I = (simulator.model.x['I'] - simulator.model.tvp['I_set']) ** 2
            cost_V = (V_coverage - target_V_coverage) ** 2
            cost = 50 * cost_I + 10 * cost_V
    else:
        # User provided custom setpoint
        # Infer cost function from non-NA setpoint values
        cost = 0
        for state_name in ['S', 'E', 'I', 'R', 'V']:
            if state_name in setpoint and not np.all(setpoint[state_name] == 0):
                cost += (simulator.model.x[state_name] - simulator.model.tvp[state_name + '_set']) ** 2

    # Update the simulator set-point
    simulator.update_dict(setpoint, name='setpoint')

    # Set cost function
    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # Set input penalty: balance control effort vs state tracking
    simulator.mpc.set_rterm(u1=rterm, u2=rterm, u3=rterm)

    # Set bounds on states and controls
    simulator.mpc.bounds['lower', '_x', 'S'] = 0.0
    simulator.mpc.bounds['lower', '_x', 'E'] = 0.0
    simulator.mpc.bounds['lower', '_x', 'I'] = 0.0
    simulator.mpc.bounds['lower', '_x', 'R'] = 0.0
    simulator.mpc.bounds['lower', '_x', 'V'] = 0.0
    
    # Control bounds
    simulator.mpc.bounds['lower', '_u', 'u1'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u1'] = 1.0   # u1 is a proportion (0-1)
    simulator.mpc.bounds['lower', '_u', 'u2'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u2'] = 0.3   # u2 is vaccination rate
    simulator.mpc.bounds['lower', '_u', 'u3'] = 0.0
    simulator.mpc.bounds['upper', '_u', 'u3'] = 0.2   # u3 is treatment rate

    # Initial condition: small outbreak in mostly susceptible population
    x0 = {
        'S': 1000.0,  # Susceptible
        'E': 500.0,    # Exposed
        'I': 100.0,    # Infected
        'R': 0.0,      # Recovered
        'V': 0.0       # Vaccinated
    }

    # Run simulation using MPC
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=x0, u=None, mpc=True, return_full_output=True)

    # Return
    return t_sim, x_sim, u_sim, y_sim, simulator


def package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim):
    """Convert simulation outputs to pandas DataFrame"""
    df_x = pd.DataFrame(x_sim)
    df_u = pd.DataFrame(u_sim)
    df_y = pd.DataFrame(y_sim)
    df_t = pd.DataFrame({'time': t_sim})
    
    # Rename measurement columns to avoid conflicts
    new_names = {key: 'sensor_' + key for key in df_y}
    df_y = df_y.rename(columns=new_names)
    
    # Merge into a single data frame
    df_trajec = pd.concat([df_t, df_x, df_u, df_y], axis=1)
    
    return df_trajec


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":
    # Create dynamics and measurement functions
    f = F(beta=0.0005, mu=0.02, c=0.2, r=0.05, F_rate=10.0)
    h = H('h_epidemic_key_indicators')
    
    # Run simulation
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_seirv(
        f.f, h.h, 
        tsim_length=300, 
        dt=1.0,
        control_strategy='vaccination_focus',
        rterm=1e-2
    )
    
    # Package as DataFrame
    df = package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot states (left column)
    axes[0, 0].plot(t_sim, x_sim['S'], label='Susceptible')
    axes[0, 0].plot(t_sim, x_sim['E'], label='Exposed')
    axes[0, 0].plot(t_sim, x_sim['I'], label='Infected')
    axes[0, 0].plot(t_sim, x_sim['R'], label='Recovered')
    axes[0, 0].plot(t_sim, x_sim['V'], label='Vaccinated')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_title('SEIR-V Model States')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot controls (right column, top)
    axes[0, 1].plot(t_sim, u_sim['u1'], label='u1 (Transmission reduction)')
    axes[0, 1].plot(t_sim, u_sim['u2'], label='u2 (Vaccination rate)')
    axes[0, 1].plot(t_sim, u_sim['u3'], label='u3 (Treatment rate)')
    axes[0, 1].set_ylabel('Control Input')
    axes[0, 1].set_title('Control Inputs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot total population (middle left)
    total_pop = x_sim['S'] + x_sim['E'] + x_sim['I'] + x_sim['R'] + x_sim['V']
    axes[1, 0].plot(t_sim, total_pop, label='Total Population')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_title('Total Population (Should approach F/μ = {:.1f})'.format(F/0.02))
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot vaccination coverage (middle right)
    vaccination_coverage = x_sim['V'] / total_pop
    axes[1, 1].plot(t_sim, vaccination_coverage, label='Vaccination Coverage', color='green')
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_title('Vaccination Coverage (V/N)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot new infections (bottom left)
    S = x_sim['S']
    I = x_sim['I']
    u1 = u_sim['u1']
    new_infections = 0.0005 * (1 - u1) * S * I
    axes[2, 0].plot(t_sim, new_infections, label='New Infections per day', color='red')
    axes[2, 0].set_xlabel('Time (days)')
    axes[2, 0].set_ylabel('New Infections')
    axes[2, 0].set_title('Daily New Infections')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Plot basic reproduction number dynamics (bottom right)
    # R0_t = β(1-u1)S / (μ+r+u3)
    effective_beta = 0.0005 * (1 - u1)
    R0_t = effective_beta * S / (0.02 + 0.05 + u_sim['u3'])
    axes[2, 1].plot(t_sim, R0_t, label='Effective R_t', color='purple')
    axes[2, 1].axhline(y=1.0, color='black', linestyle='--', label='R=1 threshold')
    axes[2, 1].set_xlabel('Time (days)')
    axes[2, 1].set_ylabel('R_t')
    axes[2, 1].set_title('Effective Reproduction Number R_t')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Susceptible: {x_sim['S'][-1]:.1f}")
    print(f"Exposed: {x_sim['E'][-1]:.1f}")
    print(f"Infected: {x_sim['I'][-1]:.1f}")
    print(f"Recovered: {x_sim['R'][-1]:.1f}")
    print(f"Vaccinated: {x_sim['V'][-1]:.1f}")
    print(f"Total Population: {total_pop[-1]:.1f}")
    print(f"Vaccination Coverage: {vaccination_coverage[-1]:.2%}")
    print(f"Effective R_t: {R0_t[-1]:.2f}")
    
    # Control usage statistics
    print("\nControl Usage Statistics:")
    print(f"Mean u1 (transmission reduction): {np.mean(u_sim['u1']):.3f}")
    print(f"Mean u2 (vaccination rate): {np.mean(u_sim['u2']):.3f}")
    print(f"Mean u3 (treatment rate): {np.mean(u_sim['u3']):.3f}")
