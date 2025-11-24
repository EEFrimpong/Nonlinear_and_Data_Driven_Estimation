import numpy as np

############################################################################################
# GLOBAL PARAMETERS
############################################################################################
mu = 0.02 / 365
sigma_default = 1.0 / 5.2
gamma = 1.0 / 10.0
N = 1_000_000

beta0_default   = 0.5
epsilon_default = 0.2
T_default       = 365


############################################################################################
# DYNAMICS CLASS F(x,u)
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

    def f(self, x_vec, u_vec, return_state_names=False):

        if return_state_names:
            return ['S','E','I','R','beta_eff','sigma','t','C']

        S        = x_vec[0]
        E        = x_vec[1]
        I        = x_vec[2]
        R        = x_vec[3]
        beta_eff = x_vec[4]
        sigma    = x_vec[5]
        t        = x_vec[6]
        C        = x_vec[7]

        u1 = u_vec[0]
        u2 = u_vec[1]
        u3 = u_vec[2]

        lambda_inf = beta_eff * (1 - u1) * S * I / self.N

        dS_dt = self.mu * self.N - lambda_inf - u2*S - self.mu*S
        dE_dt = lambda_inf - sigma*E - self.mu*E
        dI_dt = sigma*E - (self.gamma+u3)*I - self.mu*I
        dR_dt = (self.gamma+u3)*I + u2*S - self.mu*R

        dbeta_dt = (
            self.beta0 *
            (-self.epsilon) * (2*np.pi/self.T) *
            np.sin(2*np.pi*t/self.T) *
            (1 - u1)
        )

        dsigma_dt = 0.0
        dt_dt     = 1.0
        dC_dt     = lambda_inf

        return np.array([dS_dt,dE_dt,dI_dt,dR_dt,dbeta_dt,dsigma_dt,dt_dt,dC_dt])


############################################################################################
# MEASUREMENT CLASS H(x,u)
############################################################################################
class H(object):
    def __init__(self, measurement_option, mu=mu, gamma=gamma, N=N):
        self.measurement_option = measurement_option
        self.mu = mu
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        func = self.__getattribute__(self.measurement_option)
        return func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    ###########################################################################
    # MAIN OBSERVABLE MEASUREMENT
    ###########################################################################
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I_measured','new_cases','Cumulative_cases','beta_eff']

        S        = x_vec[0]
        I        = x_vec[2]
        beta_eff = x_vec[4]
        C        = x_vec[7]

        u1 = u_vec[0]

        new_cases = beta_eff*(1-u1)*S*I/self.N

        return np.array([I, new_cases, C, beta_eff])
