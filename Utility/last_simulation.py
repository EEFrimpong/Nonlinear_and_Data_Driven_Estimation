import numpy as np

############################################################################################
# Global parameters (tweak as you like)
############################################################################################
mu     = 0.02 / 365      # Natural mortality rate per day (2% per year)
sigma  = 1.0 / 5.2       # Progression rate from E to I
gamma  = 1.0 / 10.0      # Recovery rate
N      = 1_000_000       # Total population

############################################################################################
# Continuous time dynamics function for [S, E, I, R, beta]
############################################################################################
class F(object):
    def __init__(self, mu=mu, sigma=sigma, gamma=gamma, N=N):
        """
        States:
            x = [S, E, I, R, beta]

        Controls:
            u1 = u_vec[0]   social distancing (transmission reduction)
            u2 = u_vec[1]   vaccination (S -> R)
            u3 = u_vec[2]   treatment (extra recovery of I)

        ODEs (your model):
            dS/dt    = mu N - beta (1 - u1) S I / N - u2 S - mu S
            dE/dt    = beta (1 - u1) S I / N - sigma E - mu E
            dI/dt    = sigma E - (gamma + u3) I - mu I
            dR/dt    = (gamma + u3) I + u2 S - mu R
            dβ/dt    = 0
        """
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, x_vec, u_vec, return_state_names=False):
        if return_state_names:
            return ['S', 'E', 'I', 'R', 'beta']

        # Extract states
        S, E, I, R, beta = x_vec

        # Extract controls
        u1 = float(u_vec[0])
        u2 = float(u_vec[1])
        u3 = float(u_vec[2])

        # Force of infection
        lambda_inf = beta * (1.0 - u1) * S * I / self.N

        # Dynamics (exactly your equations)
        dS_dt    = self.mu * self.N - lambda_inf - u2 * S - self.mu * S
        dE_dt    = lambda_inf - self.sigma * E - self.mu * E
        dI_dt    = self.sigma * E - (self.gamma + u3) * I - self.mu * I
        dR_dt    = (self.gamma + u3) * I + u2 * S - self.mu * R
        dbeta_dt = 0.0

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt, dbeta_dt])


############################################################################################
# Continuous time measurement functions (same structure as your H class)
############################################################################################
class H(object):
    def __init__(self, measurement_option,
                 mu=mu, sigma=sigma, gamma=gamma, N=N):
        """
        measurement_option: string naming which h_* function to use.
        """
        self.measurement_option = measurement_option
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = getattr(self, self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # -------------------------------------------------------------------------
    # 1. h_i_only: I
    # -------------------------------------------------------------------------
    def h_i_only(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I]
        """
        if return_measurement_names:
            return ['I_measured']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 2. h_ir: I, R
    # -------------------------------------------------------------------------
    def h_ir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, R]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured']
        I = x_vec[2]
        R = x_vec[3]
        return np.array([I, R])

    # -------------------------------------------------------------------------
    # 3. h_reported_cases: I (reported infectious)
    # -------------------------------------------------------------------------
    def h_reported_cases(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I]
        """
        if return_measurement_names:
            return ['I_reported']
        I = x_vec[2]
        return np.array([I])

    # -------------------------------------------------------------------------
    # 4. h_incidence: I, new_cases
    # new_cases = beta (1 - u1) S I / N
    # -------------------------------------------------------------------------
    def h_incidence(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, new_cases]^T
        """
        if return_measurement_names:
            return ['I_reported', 'new_cases']

        S    = x_vec[0]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, new_cases])

    # -------------------------------------------------------------------------
    # 5. h_incidence_recovery: I, R, new_cases
    # -------------------------------------------------------------------------
    def h_incidence_recovery(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, R, new_cases]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases']

        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_cases = beta * (1.0 - u1) * S * I / self.N
        return np.array([I, R, new_cases])

    # -------------------------------------------------------------------------
    # 6. h_ei_flows: E, I, new_inf, prog
    # new_inf = beta (1 - u1) S I / N
    # prog    = sigma E
    # -------------------------------------------------------------------------
    def h_ei_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [E, I, new_inf, prog]^T
        """
        if return_measurement_names:
            return ['E_measured', 'I_measured', 'new_inf', 'prog']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E

        return np.array([E, I, new_inf, prog])

    # -------------------------------------------------------------------------
    # 7. h_seir: S, E, I, R
    # -------------------------------------------------------------------------
    def h_seir(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R]^T
        """
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
        """
        Measurement: y = [S, E, I, R, new_inf]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]
        u1   = u_vec[0]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        return np.array([S, E, I, R, new_inf])

    # -------------------------------------------------------------------------
    # 9. h_seir_with_beta: S, E, I, R, beta
    # -------------------------------------------------------------------------
    def h_seir_with_beta(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [S, E, I, R, beta]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'beta_measured']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        return np.array([S, E, I, R, beta])

    # -------------------------------------------------------------------------
    # 10. h_with_flows: S, E, I, R, new_inf, prog, recov
    # new_inf = beta (1 - u1) S I / N
    # prog    = sigma E
    # recov   = (gamma + u3) I
    # -------------------------------------------------------------------------
    def h_with_flows(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement:
            y = [S, E, I, R, new_inf, prog, recov]^T
        """
        if return_measurement_names:
            return ['S_measured', 'E_measured', 'I_measured', 'R_measured',
                    'new_inf', 'prog', 'recov']

        S    = x_vec[0]
        E    = x_vec[1]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        u1 = u_vec[0]
        u3 = u_vec[2]

        new_inf = beta * (1.0 - u1) * S * I / self.N
        prog    = self.sigma * E
        recov   = (self.gamma + u3) * I

        return np.array([S, E, I, R, new_inf, prog, recov])

    # -------------------------------------------------------------------------
    # 11. h_observable: I, R, new_cases, recov
    # (adapted: no cumulative C in this simpler model)
    # -------------------------------------------------------------------------
    def h_observable(self, x_vec, u_vec, return_measurement_names=False):
        """
        Measurement: y = [I, R, new_cases, recov]^T
        """
        if return_measurement_names:
            return ['I_measured', 'R_measured', 'new_cases', 'recov']

        S    = x_vec[0]
        I    = x_vec[2]
        R    = x_vec[3]
        beta = x_vec[4]

        u1 = u_vec[0]
        u3 = u_vec[2]

        new_cases = beta * (1.0 - u1) * S * I / self.N
        recov     = (self.gamma + u3) * I

        return np.array([I, R, new_cases, recov])


############################################################################################
# Tiny usage example (you can plug this into your pybounds simulator)
############################################################################################
if __name__ == "__main__":
    f = F()
    h = H(measurement_option='h_with_flows')

    # Example state and control
    x_example = np.array([0.7*N, 0.05*N, 0.01*N, 0.24*N, 0.3])
    u_example = np.array([0.2, 0.1, 0.0])

    dx = f.f(x_example, u_example)
    y  = h.h(x_example, u_example)

    print("dx/dt =", dx)
    print("measurement y =", y)
