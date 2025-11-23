# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate

import pybounds


# Epidemiological parameters (replace drone parameters)


mu    = 0.0000414       # birth/death rate (example, per year)
beta  = 0.5            # transmission rate
sigma = 0.2        # exposed -> infectious rate
gamma = 0.4       # recovery rate
N     = 1_000_000      # total population (assumed constant)



class F(object):
    def __init__(self, k=None):
        """
        SEIR control-affine dynamics with state x = [S, E, I, R, k]^T.
        If k is provided (not None), then the state is [S, E, I, R] and
        k is treated as a fixed parameter.
        """
        self.k = k

    def f(self, x_vec, u_vec, mu=mu, beta=beta, sigma=sigma, gamma=gamma, N=N,
          return_state_names=False):
        """
        Continuous time SEIR dynamics in control-affine form:

        x_dot = f0(x) + f1(x) * u1 + f2(x) * u2 + f3(x) * u3
        """

        k = self.k

        # Dimension checks
        if x_vec is not None:
            if k is None:
                assert len(x_vec) == 5
            else:
                assert len(x_vec) == 4

        # Return state names if requested
        if return_state_names:
            if k is None:
                return ['S', 'E', 'I', 'R', 'k']
            else:
                return ['S', 'E', 'I', 'R']

        # Extract states
        S = x_vec[0]
        E = x_vec[1]
        I_ = x_vec[2]
        R = x_vec[3]

        if k is None:
            k = x_vec[4]

        # Extract control inputs
        u1 = u_vec[0]
        u2 = u_vec[1]
        u3 = u_vec[2]

        # Common infection term
        infection_term = beta * S * I_ / N

        # f0(x): drift dynamics
        f0_contribution = np.array([
            mu * N - infection_term - mu * S,      # dS/dt
            infection_term - (sigma + mu) * E,     # dE/dt
            sigma * E - (gamma + mu) * I_,         # dI/dt
            gamma * I_ - mu * R,                   # dR/dt
            0.0                                    # dk/dt
        ])

        # f1(x) * u1
        f1_contribution = u1 * np.array([
            infection_term,
            -infection_term,
            0.0,
            0.0,
            0.0
        ])

        # f2(x) * u2
        f2_contribution = u2 * np.array([
            -S,
            0.0,
            0.0,
            S,
            0.0
        ])

        # f3(x) * u3
        f3_contribution = u3 * np.array([
            0.0,
            0.0,
            -I_,
            I_,
            0.0
        ])

        # Combined dynamics
        x_dot_vec = f0_contribution + f1_contribution + f2_contribution + f3_contribution

        if self.k is None:
            return x_dot_vec
        else:
            return x_dot_vec[0:4]


# Measurement function y = [I, R]


class H(object):
    def __init__(self, measurement_option='h_I_R', k=None):
        self.k = k
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    def h_I_R(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['I', 'R']

        S = x_vec[0]
        E = x_vec[1]
        I_ = x_vec[2]
        R = x_vec[3]

        return np.array([I_, R])


# Simulation wrapper using pybounds (MPC form)


def simulate_seir(f, h, tsim_length=365, dt=1.0, measurement_names=None, rterm=1e-4):

    # Set names
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2', 'u3']

    # Get measurement names
    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    simulator = pybounds.Simulator(
        f, h, dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(10/dt)
    )

    # Time grid
    tsim = np.arange(0, tsim_length, step=dt)
    NA = np.zeros_like(tsim)

    # Desired targets
    I_set = np.zeros_like(tsim)
    R_set = np.zeros_like(tsim)

    setpoint = {
        'S': NA,
        'E': NA,
        'I': I_set,
        'R': R_set,
        'k': np.ones_like(tsim),
    }

    if 'k' not in state_names:
        del setpoint['k']

    simulator.update_dict(setpoint, name='setpoint')

    # Cost function
    I_error = simulator.model.x['I'] - simulator.model.tvp['I_set']
    R_error = simulator.model.x['R'] - simulator.model.tvp['R_set']
    cost = I_error**2 + R_error**2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)
    simulator.mpc.set_rterm(u1=0, u2=0, u3=0)

    # Simulate
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=None, u=None, mpc=True, return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


# Helper function (unchanged formatting only)


def generate_smooth_curve(t_points, method='spline', smoothness=0.1,
                          amplitude=1.0, seed=None):

    if seed is not None:
        rng = np.random.default_rng(seed)

    t_points = np.array(t_points)

    if method == 'spline':
        n_control = max(5, int(len(t_points) * smoothness))
        control_t = np.linspace(t_points[0], t_points[-1], n_control)
        control_y = np.random.normal(0, amplitude/3, n_control)

        spline = interpolate.CubicSpline(control_t, control_y)
        return spline(t_points)

    elif method == 'sine_sum':
        n_harmonics = max(3, int(20 * smoothness))
        result = np.zeros_like(t_points, dtype=float)

        for i in range(n_harmonics):
            freq = np.random.exponential(1.0 / smoothness)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0, amplitude) / (i + 1)
            result += amp * np.sin(2 * np.pi * freq * t_points + phase)

        return result

    elif method == 'noise_filter':
        from scipy.signal import butter, filtfilt

        noise = np.random.normal(0, amplitude, len(t_points))
        nyquist = 0.5 * len(t_points) / (t_points[-1] - t_points[0])
        cutoff = nyquist * smoothness
        b, a = butter(3, cutoff / nyquist, btype='low')

        return filtfilt(b, a, noise)

    else:
        raise ValueError("Method must be 'spline', 'sine_sum', or 'noise_filter'")
