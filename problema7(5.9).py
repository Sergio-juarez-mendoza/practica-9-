import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4_system(f1, f2, t0, x10, x20, h, steps, k1, k2, k3, k4):
    """
    Método de Runge-Kutta de 4to orden para el sistema Lotka-Volterra.

    Parámetros:
    - f1, f2: Funciones que definen las EDOs (dx1/dt y dx2/dt).
    - t0: Tiempo inicial.
    - x10, x20: Poblaciones iniciales de presas y depredadores.
    - h: Tamaño del paso.
    - steps: Número de pasos.
    - k1, k2, k3, k4: Constantes del modelo.

    Retorna:
    - t: Array de tiempos.
    - x1, x2: Arrays de poblaciones de presas y depredadores.
    """
    t = np.zeros(steps + 1)
    x1 = np.zeros(steps + 1)
    x2 = np.zeros(steps + 1)
    t[0], x1[0], x2[0] = t0, x10, x20

    for i in range(steps):
        # Coeficientes k para x1 (presas)
        k1_x1 = h * f1(t[i], x1[i], x2[i], k1, k2)
        k1_x2 = h * f2(t[i], x1[i], x2[i], k3, k4)

        k2_x1 = h * f1(t[i] + h / 2, x1[i] + k1_x1 / 2, x2[i] + k1_x2 / 2, k1, k2)
        k2_x2 = h * f2(t[i] + h / 2, x1[i] + k1_x1 / 2, x2[i] + k1_x2 / 2, k3, k4)

        k3_x1 = h * f1(t[i] + h / 2, x1[i] + k2_x1 / 2, x2[i] + k2_x2 / 2, k1, k2)
        k3_x2 = h * f2(t[i] + h / 2, x1[i] + k2_x1 / 2, x2[i] + k2_x2 / 2, k3, k4)

        k4_x1 = h * f1(t[i] + h, x1[i] + k3_x1, x2[i] + k3_x2, k1, k2)
        k4_x2 = h * f2(t[i] + h, x1[i] + k3_x1, x2[i] + k3_x2, k3, k4)

        # Actualización
        x1[i + 1] = x1[i] + (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1) / 6
        x2[i + 1] = x2[i] + (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2) / 6
        t[i + 1] = t[i] + h

    return t, x1, x2


# Definición de las EDOs del modelo Lotka-Volterra
def f1_presas(t, x1, x2, k1, k2):
    return k1 * x1 - k2 * x1 * x2  # dx1/dt = k1*x1 - k2*x1*x2


def f2_depredadores(t, x1, x2, k3, k4):
    return k3 * x1 * x2 - k4 * x2  # dx2/dt = k3*x1*x2 - k4*x2


# Parámetros del modelo (ejemplo)
k1 = 0.4  # Tasa de natalidad de presas
k2 = 0.01  # Tasa de mortalidad de presas por depredadores
k3 = 0.001  # Tasa de natalidad de depredadores por presas
k4 = 0.3  # Tasa de mortalidad de depredadores

# Configuración de la simulación
t0 = 0
x10 = 50  # Población inicial de presas
x20 = 20  # Población inicial de depredadores
t_final = 100
h = 0.1
steps = int((t_final - t0) / h)

# Resolver el sistema
t, x1, x2 = runge_kutta_4_system(f1_presas, f2_depredadores, t0, x10, x20, h, steps, k1, k2, k3, k4)

# Gráficas
plt.figure(figsize=(12, 6))

# Poblaciones vs tiempo
plt.subplot(1, 2, 1)
plt.plot(t, x1, 'g-', label='Presas ($x_1$)')
plt.plot(t, x2, 'r-', label='Depredadores ($x_2$)')
plt.title('Dinámica de Poblaciones (Lotka-Volterra)')
plt.xlabel('Tiempo ($t$)')
plt.ylabel('Población')
plt.legend()
plt.grid()

# Diagrama de fase
plt.subplot(1, 2, 2)
plt.plot(x1, x2, 'b-')
plt.title('Diagrama de Fase ($x_1$ vs $x_2$)')
plt.xlabel('Presas ($x_1$)')
plt.ylabel('Depredadores ($x_2$)')
plt.grid()

plt.tight_layout()
plt.show()