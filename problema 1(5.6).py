import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4_system(f1, f2, t0, y0, z0, h, steps):
    """Genera valores iniciales para AB4 usando RK4 en sistemas."""
    t = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    z = np.zeros(steps + 1)
    t[0], y[0], z[0] = t0, y0, z0

    for i in range(steps):
        k1_y = h * f1(t[i], y[i], z[i])
        k1_z = h * f2(t[i], y[i], z[i])

        k2_y = h * f1(t[i] + h / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)
        k2_z = h * f2(t[i] + h / 2, y[i] + k1_y / 2, z[i] + k1_z / 2)

        k3_y = h * f1(t[i] + h / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)
        k3_z = h * f2(t[i] + h / 2, y[i] + k2_y / 2, z[i] + k2_z / 2)

        k4_y = h * f1(t[i] + h, y[i] + k3_y, z[i] + k3_z)
        k4_z = h * f2(t[i] + h, y[i] + k3_y, z[i] + k3_z)

        y[i + 1] = y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z[i + 1] = z[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        t[i + 1] = t[i] + h

    return t, y, z


def adams_bashforth_4_system(f1, f2, t, y, z, h):
    """AB4 para sistemas de EDOs."""
    for i in range(3, len(t) - 1):
        # Predictor para y (posición)
        f_y = [f1(t[i - j], y[i - j], z[i - j]) for j in range(4)]
        y[i + 1] = y[i] + h / 24 * (55 * f_y[0] - 59 * f_y[1] + 37 * f_y[2] - 9 * f_y[3])

        # Predictor para z (velocidad)
        f_z = [f2(t[i - j], y[i - j], z[i - j]) for j in range(4)]
        z[i + 1] = z[i] + h / 24 * (55 * f_z[0] - 59 * f_z[1] + 37 * f_z[2] - 9 * f_z[3])

        t[i + 1] = t[i] + h
    return t, y, z


# Definición de los sistemas para cada problema
# Problema a: y'' = |t|^2 - 2y → Sistema: y' = z, z' = t^2 - 2y
def f1_a(t, y, z):
    return z


def f2_a(t, y, z):
    return t ** 2 - 2 * y


def y_real_a(t):
    return (1 / 3) * t ** 3 + (1 / 3) * np.exp(t)


# Problema b: y'' = 1 + (t - y)^2 → Sistema: y' = z, z' = 1 + (t - y)^2
def f1_b(t, y, z):
    return z


def f2_b(t, y, z):
    return 1 + (t - y) ** 2


def y_real_b(t):
    return t + 1 / (t - 1)


# Problema c: y'' = 1 + y/t → Sistema: y' = z, z' = 1 + y/t
def f1_c(t, y, z):
    return z


def f2_c(t, y, z):
    return 1 + y / t if t != 0 else 0


def y_real_c(t):
    return t * np.log(t) + 2 * t


# Problema d: y'' = cos(2t) + sin(3t) → Sistema: y' = z, z' = cos(2t) + sin(3t)
def f1_d(t, y, z):
    return z


def f2_d(t, y, z):
    return np.cos(2 * t) + np.sin(3 * t)


def y_real_d(t):
    return 0.5 * np.sin(2 * t) - (1 / 3) * np.cos(3 * t) + 4 / 3


# Configuración de los problemas
problemas = [
    {"f1": f1_a, "f2": f2_a, "t0": 0, "y0": 0, "z0": 1 / 3, "t_final": 1, "h": 0.2, "y_real": y_real_a,
     "nombre": "a) y'' = t^2 - 2y"},
    {"f1": f1_b, "f2": f2_b, "t0": 2, "y0": 1, "z0": 1, "t_final": 3, "h": 0.2, "y_real": y_real_b,
     "nombre": "b) y'' = 1 + (t - y)^2"},
    {"f1": f1_c, "f2": f2_c, "t0": 1, "y0": 2, "z0": 1, "t_final": 2, "h": 0.2, "y_real": y_real_c,
     "nombre": "c) y'' = 1 + y/t"},
    {"f1": f1_d, "f2": f2_d, "t0": 0, "y0": 1, "z0": 0, "t_final": 1, "h": 0.2, "y_real": y_real_d,
     "nombre": "d) y'' = cos(2t) + sin(3t)"}
]

# Resolver cada problema
for problema in problemas:
    # Generar valores iniciales con RK4
    steps_rk4 = 3  # AB4 necesita 4 puntos iniciales
    t_rk4, y_rk4, z_rk4 = runge_kutta_4_system(
        problema["f1"], problema["f2"],
        problema["t0"], problema["y0"], problema["z0"],
        problema["h"], steps_rk4
    )

    # Extender arrays para AB4
    total_steps = int((problema["t_final"] - problema["t0"]) / problema["h"])
    t = np.zeros(total_steps + 1)
    y = np.zeros(total_steps + 1)
    z = np.zeros(total_steps + 1)
    t[:4] = t_rk4
    y[:4] = y_rk4
    z[:4] = z_rk4

    # Aplicar AB4
    t, y, z = adams_bashforth_4_system(problema["f1"], problema["f2"], t, y, z, problema["h"])

    # Calcular solución real y error
    y_real = problema["y_real"](t)

    # Resultados
    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox\t y_real\t\t Error")
    for ti, yi, yri in zip(t, y, y_real):
        print(f"{ti:.2f}\t {yi:.6f}\t {yri:.6f}\t {np.abs(yi - yri):.6f}")

    # Gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='AB4 Aproximación')
    plt.plot(t, y_real, 'r-', label='Solución Real')
    plt.title(f"Comparación AB4 vs Real: {problema['nombre']}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()