import numpy as np
import matplotlib.pyplot as plt
import pybounds

############################################################################################
# Global Parameters
############################################################################################
mu = 0.02 / 365
gamma = 1.0 / 10.0
N = 1_000_000

# Initial guesses / baselines
beta0_default = 0.3
sigma_default = 1.0 / 10.2

# Beta dynamics
beta_decay_rate = 0.001


############################################################################################
# Continuous Time Dynamics
############################################################################################
class F(object):

    def __init__(self,
                 mu=mu,
                 gamma=gamma,
                 N=N,
                 beta0=beta0_default,
                 beta_decay=beta_decay_rate):

        self.mu = mu
        self.gamma = gamma
        self.N = N

        self.beta0 = beta0
        self.beta_decay = beta_decay

    def f(self, x_vec, u_vec, return_state_names=False):

        if return_state_names:
            return ['S','E','I','R','beta','sigma']

        # States
        S, E, I, R, beta, sigma = x_vec

        # Controls
        u1, u2, u3, u4 = u_vec

        # Force of infection
        lambda_inf = beta*(1-u1) * S * I / self.N

        # SEIR dynamics
        dS = self.mu*self.N - lambda_inf - u2*S - self.mu*S
        dE = lambda_inf - sigma*E - self.mu*E
        dI = sigma*E - (self.gamma + u3)*I - self.mu*I
        dR = (self.gamma + u3)*I + u2*S - self.mu*R

        # Parameter dynamics
        dbeta = self.beta_decay*(self.beta0 - beta) - 0.1*u4*beta
        dsigma = 0.0

        return np.array([dS, dE, dI, dR, dbeta, dsigma])


############################################################################################
# Measurement Models
############################################################################################
class H(object):

    def __init__(self, option, mu=mu, gamma=gamma, N=N):
        self.option = option
        self.mu = mu
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = self.__getattribute__(self.option)
        return func(x_vec, u_vec, return_measurement_names)

    def _beta_eff(self, beta, u1):
        return beta*(1-u1)

    # ---------------------------------------------------------------------
    def h_i_only(self, x, u, rmn=False):
        if rmn: return ['I']
        return np.array([x[2]])

    # ---------------------------------------------------------------------
    def h_ir(self, x, u, rmn=False):
        if rmn: return ['I','R']
        return np.array([x[2],x[3]])

    # ---------------------------------------------------------------------
    def h_seir(self, x, u, rmn=False):
        if rmn: return ['S','E','I','R']
        return x[0:4]

    # ---------------------------------------------------------------------
    def h_beta_sigma(self, x, u, rmn=False):
        if rmn: return ['beta','sigma']
        return np.array([x[4],x[5]])

    # ---------------------------------------------------------------------
    def h_incidence(self, x, u, rmn=False):
        if rmn: return ['I','new_cases']

        S,E,I,R,beta,sigma = x
        u1 = u[0]

        new_cases = self._beta_eff(beta,u1)*S*I/self.N
        return np.array([I,new_cases])

    # ---------------------------------------------------------------------
    def h_ei_flows(self, x, u, rmn=False):
        if rmn: return ['E','I','new_inf','prog']

        S,E,I,R,beta,sigma = x
        u1 = u[0]

        new_inf = self._beta_eff(beta,u1)*S*I/self.N
        prog = sigma*E

        return np.array([E,I,new_inf,prog])

    # ---------------------------------------------------------------------
    def h_full_state(self, x, u, rmn=False):
        if rmn:
            return ['S','E','I','R','beta','sigma']
        return x


############################################################################################
# Simulator Wrapper
############################################################################################
def simulate_seir(f, h,
                  tsim_length=365,
                  dt=1,
                  measurement_names=None,
                  setpoint=None,
                  x0=None,
                  measurement_noise_stds=None):

    # Initial conditions
    if x0 is None:
        x0 = np.array([
            0.7*N,
            0.05*N,
            0.05*N,
            N*(1-0.7-0.05-0.05),
            beta0_default,
            sigma_default
        ])

    state_names = f(None,None,True)
    input_names = ['u1','u2','u3','u4']

    if measurement_names is None:
        measurement_names = h(None,None,True)

    simulator = pybounds.Simulator(
        f,h,
        dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=10
    )

    if measurement_noise_stds is not None:
        simulator.measurement_noise_std = np.array([
            measurement_noise_stds.get(m,0.0) for m in measurement_names
        ])

    tsim = np.arange(0,tsim_length,dt)

    # Default setpoints
    if setpoint is None:
        setpoint = {
            'S': np.zeros_like(tsim),
            'E': 0.00005*N*np.ones_like(tsim),
            'I': 0.0001*N*np.ones_like(tsim),
            'R': np.zeros_like(tsim),
            'beta': beta0_default*np.ones_like(tsim),
            'sigma': sigma_default*np.ones_like(tsim)
        }

    simulator.update_dict(setpoint,name='setpoint')

    # MPC cost
    cost = (
        10*(simulator.model.x['E']-simulator.model.tvp['E_set'])**2
      + 100*(simulator.model.x['I']-simulator.model.tvp['I_set'])**2
      + 1*(simulator.model.x['beta']-simulator.model.tvp['beta_set'])**2
    )

    simulator.mpc.set_objective(cost,cost)
    simulator.mpc.set_rterm(u1=1e-4, u2=1e-4, u3=1e-4, u4=1e-4)

    # State bounds
    eps = 1e-6

    for s in ['S','E','I','R']:
        simulator.mpc.bounds['lower','_x',s] = eps
        simulator.mpc.bounds['upper','_x',s] = N

    simulator.mpc.bounds['lower','_x','beta']  = 0.0
    simulator.mpc.bounds['upper','_x','beta'] = 1.0

    simulator.mpc.bounds['lower','_x','sigma']  = 0.001
    simulator.mpc.bounds['upper','_x','sigma'] = 1.0

    # Controls
    simulator.mpc.bounds['lower','_u','u1'] = 0.0
    simulator.mpc.bounds['upper','_u','u1'] = 0.9
    simulator.mpc.bounds['lower','_u','u2'] = 0.0
    simulator.mpc.bounds['upper','_u','u2'] = 0.6
    simulator.mpc.bounds['lower','_u','u3'] = 0.0
    simulator.mpc.bounds['upper','_u','u3'] = 0.5
    simulator.mpc.bounds['lower','_u','u4'] = 0.0
    simulator.mpc.bounds['upper','_u','u4'] = 0.5

    # Run simulation
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=x0,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Example usage
############################################################################################
if __name__ == "__main__":

    f = F()
    h = H('h_ei_flows')

    noise = {
        'E':500,
        'I':300,
        'new_inf':400,
        'prog':200
    }

    t, x, u, y, sim = simulate_seir(
        f, h,
        tsim_length=250,
        measurement_noise_stds=noise
    )

    plt.figure(figsize=(10,6))
    plt.plot(t, x[:,2], label='I')
    plt.plot(t, x[:,1], label='E')
    plt.plot(t, x[:,4], label='beta')
    plt.plot(t, x[:,5], label='sigma')
    plt.legend()
    plt.title('SEIR with β and σ as joint states')
    plt.xlabel('Time (days)')
    plt.grid(True)
    plt.show()

