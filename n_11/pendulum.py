import numpy as np
from scipy.integrate import solve_ivp

from n_11.const import L1, L2, M1, M2, G, PI, T_MAX, T_POINTS


# ref: prezentace
def derive_positions(len_a, len_b, theta_a, theta_b):
    x_a = len_a * np.sin(theta_a)
    y_a = -len_a * np.cos(theta_a)
    x_b = len_a * np.sin(theta_a) + len_b * np.sin(theta_b)
    y_b = -len_a * np.cos(theta_a) - len_b * np.cos(theta_b)
    return x_a, y_a, x_b, y_b


# ref: prezentace (+ úpravy)
def derive_velocities(len_a, len_b, omega_a, omega_b, theta_a, theta_b):
    # omega_a je úhlová rychlost prvního ramene (dtheta_a/dt)
    # omega_b je úhlová rychlost druhého ramene (dtheta_b/dt)
    la_oa = len_a * omega_a
    v_a_x = la_oa * np.cos(theta_a)
    v_a_y = la_oa * np.sin(theta_a)

    lb_ob = len_b * omega_b
    v_b_x = v_a_x + lb_ob * np.cos(theta_b)
    v_b_y = v_a_y + lb_ob * np.sin(theta_b)
    return v_a_x, v_a_y, v_b_x, v_b_y

# diferenciální rovnice
# počítá derivace stavových proměnných: [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
# y_state: vektor stavu [theta1, omega1, theta2, omega2]
# L1, L2, M1, M2, G: parametry kyvadla
def pendulum_derivs(t, y_state, len_1, len_2, mass_1, mass_2, grav_accel):
    theta1, omega1, theta2, omega2 = y_state
    delta_theta = theta1 - theta2  # rozdíl úhlů

    # zrychlení prvního kyvadla (domega1_dt nebo theta1_ddot)
    den1 = len_1 * (mass_1 + mass_2 * np.sin(delta_theta) ** 2)
    num1 = (mass_2 * grav_accel * np.sin(theta2) * np.cos(delta_theta) -
            mass_2 * np.sin(delta_theta) * (len_1 * omega1 ** 2 * np.cos(delta_theta) + len_2 * omega2 ** 2) -
            (mass_1 + mass_2) * grav_accel * np.sin(theta1))
    domega1_dt = num1 / den1

    # zrychlení druhého kyvadla (domega2_dt nebo theta2_ddot)
    den2 = len_2 * (mass_1 + mass_2 * np.sin(delta_theta) ** 2)
    num2 = ((mass_1 + mass_2) * (
                len_1 * omega1 ** 2 * np.sin(delta_theta) - grav_accel * np.sin(theta2) + grav_accel * np.sin(
            theta1) * np.cos(delta_theta)) +
            mass_2 * len_2 * omega2 ** 2 * np.sin(delta_theta) * np.cos(delta_theta))
    domega2_dt = num2 / den2

    return [omega1, domega1_dt, omega2, domega2_dt]


def init_state():
    # úvodní úhly a úhlové rychlosti
    theta1_0 = (2 * PI) / 6
    omega1_0 = 0.0
    theta2_0 = (5 * PI) / 8
    omega2_0 = 0.0
    # [theta1, omega1, theta2, omega2]
    return [theta1_0, omega1_0, theta2_0, omega2_0]


def init_random_state():
    # náhodné úvodní úhly a úhlové rychlosti pro restart
    theta1_0 = np.random.uniform(-PI, PI)
    omega1_0 = np.random.uniform(-2, 2)
    theta2_0 = np.random.uniform(-PI, PI)
    omega2_0 = np.random.uniform(-2, 2)
    return [theta1_0, omega1_0, theta2_0, omega2_0]


def run_simulation_logic(current_y0_state):
    # řešení diferenciálních rovnic a výpočet pozic
    solution = solve_ivp(pendulum_derivs, [0, T_MAX], current_y0_state,
                         args=(L1, L2, M1, M2, G),
                         dense_output=True, t_eval=T_POINTS)

    # extrakce výsledků řešení (theta - úhly, omega - úhlové rychlosti)
    time_p = solution.t
    theta1_sol = solution.y[0]
    # omega1_sol = solution.y[1]
    theta2_sol = solution.y[2]
    # omega2_sol = solution.y[3]

    x1_s, y1_s, x2_s, y2_s = derive_positions(L1, L2, theta1_sol, theta2_sol)
    return time_p, x1_s, y1_s, x2_s, y2_s