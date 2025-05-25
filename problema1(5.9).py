import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4_system(functions, t0, initial_conditions, h, steps):
    """
    Método de Runge-Kutta de 4to orden para sistemas de EDOs.

    Parámetros:
    - functions: Lista de funciones [f1, f2, ..., fn] que definen las EDOs.
    - t0: Tiempo inicial.
    - initial_conditions: Lista de valores iniciales [u1_0, u2_0, ..., un_0].
    - h: Tamaño del paso.
    - steps: Número de pasos.

    Retorna:
    - t: Array de tiempos.
    - solutions: Matriz de soluciones (cada fila corresponde a una variable).
    """
    n = len(functions)
    t = np.zeros(steps + 1)
    solutions = np.zeros((n, steps + 1))
    t[0] = t0
    solutions[:, 0] = initial_conditions

    for i in range(steps):
        k1 = np.zeros(n)
        k2 = np.zeros(n)
        k3 = np.zeros(n)
        k4 = np.zeros(n)

        # Cálculo de k1
        args = [t[i]] + list(solutions[:, i])
        for j in range(n):
            k1[j] = h * functions[j](*args)

        # Cálculo de k2
        args_k2 = [t[i] + h / 2] + list(solutions[:, i] + k1 / 2)
        for j in range(n):
            k2[j] = h * functions[j](*args_k2)

        # Cálculo de k3
        args_k3 = [t[i] + h / 2] + list(solutions[:, i] + k2 / 2)
        for j in range(n):
            k3[j] = h * functions[j](*args_k3)

        # Cálculo de k4
        args_k4 = [t[i] + h] + list(solutions[:, i] + k3)
        for j in range(n):
            k4[j] = h * functions[j](*args_k4)

        # Actualización
        solutions[:, i + 1] = solutions[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t[i + 1] = t[i] + h

    return t, solutions


# ==============================================
# Problema (a)
# ==============================================
def f1_a(t, u1, u2):
    return 3 * u1 + 2 * u2 - (2 * t ** 2 + 1) * np.exp(2 * t)


def f2_a(t, u1, u2):
    return 4 * u1 + u2 + (t ** 2 + 2 * t - 4) * np.exp(2 * t)


def u1_real_a(t):
    return (1 / 2) * np.exp(t ** 2) - (1 / 2) * np.exp(-t) + np.exp(2 * t)


def u2_real_a(t):
    return (1 / 2) * np.exp(t ** 2) + (3 / 2) * np.exp(-t) + t ** 2 * np.exp(2 * t)


# Configuración
t0_a = 0
u0_a = [1, 1]  # u1(0) = 1, u2(0) = 1
t_final_a = 1
h_a = 0.2
steps_a = int((t_final_a - t0_a) / h_a)

# Solución
t_a, sol_a = runge_kutta_4_system([f1_a, f2_a], t0_a, u0_a, h_a, steps_a)
u1_real_a_vals = u1_real_a(t_a)
u2_real_a_vals = u2_real_a(t_a)

# Gráficas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_a, sol_a[0], 'bo-', label='RK4 $u_1(t)$')
plt.plot(t_a, u1_real_a_vals, 'r-', label='Real $u_1(t)$')
plt.title("Problema (a): $u_1(t)$")
plt.xlabel("t")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_a, sol_a[1], 'bo-', label='RK4 $u_2(t)$')
plt.plot(t_a, u2_real_a_vals, 'r-', label='Real $u_2(t)$')
plt.title("Problema (a): $u_2(t)$")
plt.xlabel("t")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==============================================
# Problema (b)
# ==============================================
def f1_b(t, u1, u2):
    return -4 * u1 - 2 * u2 + np.cos(t) + 4 * np.sin(t)


def f2_b(t, u1, u2):
    return 3 * u1 + u2 - 3 * np.sin(t)


def u1_real_b(t):
    return 2 * np.exp(-t) - 2 * np.exp(-2 * t) + np.sin(t)


def u2_real_b(t):
    return -3 * np.exp(-t) + 2 * np.exp(-2 * t)


# Configuración
t0_b = 0
u0_b = [0, -1]  # u1(0) = 0, u2(0) = -1
t_final_b = 2
h_b = 0.1
steps_b = int((t_final_b - t0_b) / h_b)

# Solución
t_b, sol_b = runge_kutta_4_system([f1_b, f2_b], t0_b, u0_b, h_b, steps_b)
u1_real_b_vals = u1_real_b(t_b)
u2_real_b_vals = u2_real_b(t_b)

# Gráficas
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_b, sol_b[0], 'bo-', label='RK4 $u_1(t)$')
plt.plot(t_b, u1_real_b_vals, 'r-', label='Real $u_1(t)$')
plt.title("Problema (b): $u_1(t)$")
plt.xlabel("t")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_b, sol_b[1], 'bo-', label='RK4 $u_2(t)$')
plt.plot(t_b, u2_real_b_vals, 'r-', label='Real $u_2(t)$')
plt.title("Problema (b): $u_2(t)$")
plt.xlabel("t")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==============================================
# Problema (c) - Sistema de 3 EDOs
# ==============================================
def f1_c(t, u1, u2, u3):
    return u2


def f2_c(t, u1, u2, u3):
    return -u1 - 2 * np.exp(t + 1)


def f3_c(t, u1, u2, u3):
    return -u1 - np.exp(t + 1)


def u1_real_c(t):
    return np.cos(t) + np.sin(t) - np.exp(t + 1)


def u2_real_c(t):
    return -np.sin(t) + np.cos(t) - np.exp(t)


def u3_real_c(t):
    return -np.sin(t) + np.cos(t)


# Configuración
t0_c = 0
u0_c = [1, 0, 1]  # u1(0)=1, u2(0)=0, u3(0)=1
t_final_c = 2
h_c = 0.5
steps_c = int((t_final_c - t0_c) / h_c)

# Solución
t_c, sol_c = runge_kutta_4_system([f1_c, f2_c, f3_c], t0_c, u0_c, h_c, steps_c)
u1_real_c_vals = u1_real_c(t_c)
u2_real_c_vals = u2_real_c(t_c)
u3_real_c_vals = u3_real_c(t_c)

# Gráficas
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(t_c, sol_c[i], 'bo-', label=f'RK4 $u_{i + 1}(t)$')
    plt.plot(t_c, [u1_real_c_vals, u2_real_c_vals, u3_real_c_vals][i], 'r-', label=f'Real $u_{i + 1}(t)$')
    plt.title(f"Problema (c): $u_{i + 1}(t)$")
    plt.xlabel("t")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()


# ==============================================
# Problema (d) - Sistema de 3 EDOs
# ==============================================
def f1_d(t, u1, u2, u3):
    return u2 - u3 + t


def f2_d(t, u1, u2, u3):
    return 3 * t ** 2


def f3_d(t, u1, u2, u3):
    return u2 + np.exp(-t)


def u1_real_d(t):
    return -0.05 * np.exp(t) + 0.25 * t ** 3 + t + 2 - np.exp(-t)


def u2_real_d(t):
    return t ** 3 + 1


def u3_real_d(t):
    return 0.25 * t ** 3 + t - np.exp(-t)


# Configuración
t0_d = 0
u0_d = [1, 1, -1]  # u1(0)=1, u2(0)=1, u3(0)=-1
t_final_d = 1
h_d = 0.1
steps_d = int((t_final_d - t0_d) / h_d)

# Solución
t_d, sol_d = runge_kutta_4_system([f1_d, f2_d, f3_d], t0_d, u0_d, h_d, steps_d)
u1_real_d_vals = u1_real_d(t_d)
u2_real_d_vals = u2_real_d(t_d)
u3_real_d_vals = u3_real_d(t_d)

# Gráficas
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(t_d, sol_d[i], 'bo-', label=f'RK4 $u_{i + 1}(t)$')
    plt.plot(t_d, [u1_real_d_vals, u2_real_d_vals, u3_real_d_vals][i], 'r-', label=f'Real $u_{i + 1}(t)$')
    plt.title(f"Problema (d): $u_{i + 1}(t)$")
    plt.xlabel("t")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()