import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
import pybounds

############################################################################################
# Global Parameters
############################################################################################
mu = 0.02 / 365             
sigma_default = 1/10.2      
gamma = 1/10.0              
N = 1_000_000

beta0_default = 0.3
epsilon_default = 0.2
T_default = 365

############################################################################################
# Continuous Time Dynamics
############################################################################################
class F(object):

    def __init__(self, mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):

        self.mu = mu
        self.gamma = gamma
        self.N = N

        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def seasonal_beta(self, t):
        return self.beta0 * (1 + self.epsilon*np.sin(2*np.pi*t/self.T))

    def f(self, x_vec, u_vec, return_state_names=False):

        if return_state_names:
            return ['S','E','I','R','sigma']

        S, E, I, R, sigma = x_vec
        
        u1, u2, u3 = u_vec

        # Simulator exposes current time automatically
        t_now = self.simulator.t_now

        beta_t = self.seasonal_beta(t_now)
        beta_eff = beta_t * (1 - u1)
        lambda_inf = beta_eff * S * I / self.N

        dS = self.mu*self.N - lambda_inf - u2*S - self.mu*S
        dE = lambda_inf - sigma*E - self.mu*E
        dI = sigma*E - (self.gamma+u3)*I - self.mu*I
        dR = (self.gamma+u3)*I + u2*S - self.mu*R

        dsigma = 0.0

        return np.array([dS,dE,dI,dR,dsigma])
        

############################################################################################
# Continuous Time Measurement Model
############################################################################################
class H(object):

    def __init__(self, option,
                 mu=mu, gamma=gamma, N=N,
                 beta0=beta0_default, epsilon=epsilon_default, T=T_default):

        self.option = option

        self.mu = mu
        self.gamma = gamma
        self.N = N

        self.beta0 = beta0
        self.epsilon = epsilon
        self.T = T

    def seasonal_beta(self, t):
        return self.beta0 * (1 + self.epsilon*np.sin(2*np.pi*t/self.T))

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = self.__getattribute__(self.option)
        return func(x_vec, u_vec, return_measurement_names)

    # ------------------------------------------------------------------
    def h_i_only(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I']
        return np.array([x_vec[2]])

    # ------------------------------------------------------------------
    def h_ir(self,x,u,return_measurement_names=False):
        if return_measurement_names:
            return ['I','R']
        return np.array([x[2],x[3]])

    # ------------------------------------------------------------------
    def h_incidence(self,x,u,return_measurement_names=False):

        if return_measurement_names:
            return ['I','new_cases']

        S,E,I,R,sigma = x

        t_now = self.simulator.t_now
        beta_eff = self.seasonal_beta(t_now)*(1-u[0])

        new_cases = beta_eff*S*I/self.N

        return np.array([I,new_cases])

    # ------------------------------------------------------------------
    def h_ei_flows(self,x,u,return_measurement_names=False):

        if return_measurement_names:
            return ['E','I','new_inf','prog']

        S,E,I,R,sigma = x

        t_now = self.simulator.t_now
        beta_eff = self.seasonal_beta(t_now)*(1-u[0])

        new_inf = beta_eff*S*I/self.N
        prog = sigma*E

        return np.array([E,I,new_inf,prog])

    # ------------------------------------------------------------------
    def h_seir(self,x,u,return_measurement_names=False):
        if return_measurement_names:
            return ['S','E','I','R']
        return x[0:4]

############################################################################################
# SEIR Simulation with MPC
############################################################################################
def simulate_seir(f,h,
                  tsim_length=365,
                  dt=1,
                  measurement_names=None,
                  setpoint=None,
                  x0=None):

    if x0 is None:
        S0=0.7*N
        E0=0.05*N
        I0=0.05*N
        R0=N-S0-E0-I0
        sigma0=sigma_default

        x0=np.array([S0,E0,I0,R0,sigma0])

    state_names=f(None,None,True)
    input_names=['u1','u2','u3']

    if measurement_names is None:
        measurement_names=h(None,None,True)

    simulator = pybounds.Simulator(
        f,h,dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=10
    )

    # Register simulator time access
    f.simulator = simulator
    h.simulator = simulator

    tsim=np.arange(0,tsim_length,dt)

    # Targets for E,I only
    I_set=0.0001*N*np.ones_like(tsim)
    E_set=0.00005*N*np.ones_like(tsim)

    setpoint={
        'S':np.zeros_like(tsim),
        'E':E_set,
        'I':I_set,
        'R':np.zeros_like(tsim),
        'sigma':sigma_default*np.ones_like(tsim)
    }

    simulator.update_dict(setpoint,name='setpoint')

    # Cost
    cost=(simulator.model.x['E']-simulator.model.tvp['E_set'])**2 + \
         (simulator.model.x['I']-simulator.model.tvp['I_set'])**2

    simulator.mpc.set_objective(cost,cost)
    simulator.mpc.set_rterm(u1=1e-4,u2=1e-4,u3=1e-4)

    # State bounds
    eps=1e-6

    for s in ['S','E','I','R']:
        simulator.mpc.bounds['lower','_x',s]=eps
        simulator.mpc.bounds['upper','_x',s]=N

    simulator.mpc.bounds['lower','_x','sigma']=0.0
    simulator.mpc.bounds['upper','_x','sigma']=1.0

    # Control bounds
    simulator.mpc.bounds['lower','_u','u1']=0
    simulator.mpc.bounds['upper','_u','u1']=0.9

    simulator.mpc.bounds['lower','_u','u2']=0
    simulator.mpc.bounds['upper','_u','u2']=0.6

    simulator.mpc.bounds['lower','_u','u3']=0
    simulator.mpc.bounds['upper','_u','u3']=0.5

    t_sim,x_sim,u_sim,y_sim = simulator.simulate(
        x0=x0,
        mpc=True,
        return_full_output=True
    )

    return t_sim,x_sim,u_sim,y_sim,simulator
