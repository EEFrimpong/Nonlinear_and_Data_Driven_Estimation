import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize
from scipy import interpolate
import pandas as pd
import pybounds

############################################################################################
# Global parameters
############################################################################################
m1 = 1.0
m2 = 1.0
k1 = 10.0
k2 = 5.0
alpha = 0.5

############################################################################################
# Dynamics model
############################################################################################
class F(object):
    def __init__(self, m1=m1, m2=m2, k1=k1, k2=k2, alpha=alpha):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha

    def f(self, x_vec, u_vec, return_state_names=False):
        """
        x = [x1, x1_dot, x2, x2_dot]
        u = [u1, u2]
        """
        if return_state_names:
            return ['x1', 'x1_dot', 'x2', 'x2_dot']

        x1, x1_dot, x2, x2_dot = x_vec
        u1, u2 = u_vec

        disp = x2 - x1
        
        linear = self.k2 * disp
        nonlinear = self.alpha * disp**3

        x1_ddot = (-self.k1 * x1 + linear + nonlinear + u1) / self.m1
        x2_ddot = (-linear - nonlinear + u2) / self.m2

        return np.array([x1_dot, x1_ddot, x2_dot, x2_ddot])


############################################################################################
# Measurements
############################################################################################
class H(object):
    def __init__(self, measurement_option='h_full_state'):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names=False):
        return getattr(self, self.measurement_option)(
            x_vec, u_vec, return_measurement_names=return_measurement_names
        )

    def h_positions(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1', 'x2']
        return np.array([x_vec[0], x_vec[2]])

    def h_velocities(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1_dot', 'x2_dot']
        return np.array([x_vec[1], x_vec[3]])

    def h_relative_position(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1', 'x2', 'x2_minus_x1']
        return np.array([x_vec[0], x_vec[2], x_vec[2] - x_vec[0]])

    def h_full_state(self, x_vec, u_vec, return_measurement_names=False):
        if return_measurement_names:
            return ['x1', 'x1_dot', 'x2', 'x2_dot']
        return x_vec


############################################################################################
# Mass-spring MPC simulator
############################################################################################
def simulate_mass_spring(f, h, tsim_length=20, dt=0.01,
                          trajectory_shape='sinusoidal',
                          measurement_names=None,
                          measurement_noise_stds=None,
                          setpoint=None,
                          rterm=1e-4):

    # ---- names ----
    state_names = f(None, None, return_state_names=True)
    input_names = ['u1', 'u2']

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    # ---- Simulator (FOLLOW DRONE PATTERN) ----
    simulator = pybounds.Simulator(
        f,
        h,
        dt=dt,
        state_names=state_names,
        input_names=input_names,
        measurement_names=measurement_names,
        mpc_horizon=int(1 / dt)
    )

    # ---- Add noise AFTER init ----
    if measurement_noise_stds is not None:
        simulator.measurement_noise_stds.update(measurement_noise_stds)

    # ---- time vector ----
    tsim = np.arange(0, tsim_length, dt)
    NA = np.zeros_like(tsim)

    # ---- Setpoint definitions ----
    if setpoint is None:
        if trajectory_shape == 'sinusoidal':
            setpoint = {
                'x1': 0.5 * np.sin(2*np.pi*0.2 * tsim),
                'x1_dot': NA,
                'x2': 0.3 * np.sin(2*np.pi*0.3 * tsim + np.pi/4),
                'x2_dot': NA
            }

        elif trajectory_shape == 'step':
            x1_ref = np.zeros_like(tsim)
            x2_ref = np.zeros_like(tsim)
            x1_ref[int(len(tsim)/4):] = 0.5
            x2_ref[int(len(tsim)/2):] = 0.3

            setpoint = {
                'x1': x1_ref,
                'x1_dot': NA,
                'x2': x2_ref,
                'x2_dot': NA
            }

        elif trajectory_shape == 'tracking':
            setpoint = {
                'x1': 0.4 * np.cos(2*np.pi*0.15 * tsim),
                'x1_dot': NA,
                'x2': 0.6 * np.cos(2*np.pi*0.15 * tsim) + 0.2,
                'x2_dot': NA
            }

        elif trajectory_shape == 'oscillating':
            setpoint = {
                'x1': 0.3*np.sin(2*np.pi*0.3*tsim),
                'x1_dot': NA,
                'x2': 0.5*np.sin(2*np.pi*0.2*tsim)
                       + 0.2*np.cos(2*np.pi*0.5*tsim),
                'x2_dot': NA
            }

    # ---- push setpoints ----
    simulator.update_dict(setpoint, name='setpoint')

    # ---- MPC objective ----
    cost_x1 = (simulator.model.x['x1']
               - simulator.model.tvp['x1_set'])**2

    cost_x2 = (simulator.model.x['x2']
               - simulator.model.tvp['x2_set'])**2

    cost = cost_x1 + cost_x2

    simulator.mpc.set_objective(mterm=cost, lterm=cost)

    # ---- input penalty ----
    simulator.mpc.set_rterm(u1=rterm, u2=rterm)

    # ---- bounds ----
    simulator.mpc.bounds['lower', '_x', 'x1'] = -2
    simulator.mpc.bounds['upper', '_x', 'x1'] =  2

    simulator.mpc.bounds['lower', '_x', 'x2'] = -2
    simulator.mpc.bounds['upper', '_x', 'x2'] =  2

    simulator.mpc.bounds['lower', '_u', 'u1'] = -50
    simulator.mpc.bounds['upper', '_u', 'u1'] =  50

    simulator.mpc.bounds['lower', '_u', 'u2'] = -50
    simulator.mpc.bounds['upper', '_u', 'u2'] =  50

    # ---- run simulation ----
    t_sim, x_sim, u_sim, y_sim = simulator.simulate(
        x0=None,
        u=None,
        mpc=True,
        return_full_output=True
    )

    return t_sim, x_sim, u_sim, y_sim, simulator


############################################################################################
# Data formatter
############################################################################################
def package_data_as_pandas_dataframe(t_sim, x_sim, u_sim, y_sim):

    df_x = pd.DataFrame(x_sim)
    df_u = pd.DataFrame(u_sim)
    df_y = pd.DataFrame(y_sim)
    df_t = pd.DataFrame({'time': t_sim})

    df_y = df_y.rename(columns={c: f'sensor_{c}' for c in df_y.columns})

    df_all = pd.concat([df_t, df_x, df_u, df_y], axis=1)

    return df_all


############################################################################################
# MAIN
############################################################################################
if __name__ == "__main__":

    # Dynamics + measurement model
    f = F()
    h = H('h_full_state')

    # Example noise (optional)
    measurement_noise_stds = {
        'x1': 0.05,
        'x2': 0.05,
        'x1_dot': 0.02,
        'x2_dot': 0.02
    }

    # Run simulation
    t_sim, x_sim, u_sim, y_sim, simulator = simulate_mass_spring(
        f.f,
        h.h,
        tsim_length=20,
        dt=0.01,
        trajectory_shape='sinusoidal',
        rterm=1e-3,
        measurement_noise_stds=measurement_noise_stds
    )

    # Package into DataFrame
    df = package_data_as_pandas_dataframe(
        t_sim, x_sim, u_sim, y_sim
    )

    # Print results
    print("\n✅ Simulation complete!")
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Plot example traj
    plt.figure(figsize=(10,4))
    plt.plot(df['time'], df['x1'], label='x1')
    plt.plot(df['time'], df['x2'], label='x2')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('2–Mass Nonlinear Spring MPC Tracking')
    plt.legend()
    plt.tight_layout()
    plt.show()
