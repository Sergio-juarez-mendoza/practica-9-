import numpy as np
import matplotlib.pyplot as plt


def rkf45(f, t0, y0, t_final, hmax, hmin, tol):
    """
    Método de Runge-Kutta-Fehlberg (RKF45) para resolver EDOs con control adaptativo del paso.

    Parámetros:
    - f: Función dy/dt = f(t, y)
    - t0: Tiempo inicial
    - y0: Valor inicial de y
    - t_final: Tiempo final
    - hmax: Tamaño máximo del paso
    - hmin: Tamaño mínimo del paso
    - tol: Tolerancia permitida

    Retorna:
    - t_valores: Array de valores de t
    - y_valores: Array de valores aproximados de y
    """
    t_valores = [t0]
    y_valores = [y0]
    h = hmax  # Paso inicial

    t = t0
    y = y0

    while t < t_final:
        if t + h > t_final:
            h = t_final - t

        # Coeficientes de RKF45
        k1 = h * f(t, y)
        k2 = h * f(t + h / 4, y + k1 / 4)
        k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = h * f(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
        k5 = h * f(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
        k6 = h * f(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)

        # Estimaciones de orden 4 y 5
        y4 = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
        y5 = y + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55

        # Estimación del error
        error = np.abs(y5 - y4)

        # Control del paso
        if error <= tol:
            t += h
            y = y4
            t_valores.append(t)
            y_valores.append(y)

        # Ajuste del paso
        if error != 0:
            h = min(hmax, max(hmin, 0.84 * h * (tol / error) ** 0.25))
        else:
            h = hmax

    return np.array(t_valores), np.array(y_valores)


# Definición de las EDOs para cada problema
def f_a(t, y):
    return y * (t - (y / t) ** 2) if t != 0 else 0


def f_b(t, y):
    return 1 + (t - y) ** 2


def f_c(t, y):
    return 1 + y / t if t != 0 else 0


def f_d(t, y):
    return np.cos(2 * t) + np.sin(3 * t)


# Soluciones reales para comparación
def y_real_a(t):
    return 0.5 * np.exp(t) - (1 / 3) * np.exp(t) + (1 / 3) * np.exp(-2 * t)


def y_real_b(t):
    return t + 1 / (1 - t)


def y_real_c(t):
    return t * np.log(t) + 2 * t


def y_real_d(t):
    return 0.5 * np.sin(2 * t) - (1 / 3) * np.cos(3 * t) + 4 / 3


# Configuración de los problemas
problemas = [
    {"f": f_a, "t0": 0, "y0": 0, "t_final": 1, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_a,
     "nombre": "a) y' = y(t - (y/t)^2)"},
    {"f": f_b, "t0": 2, "y0": 1, "t_final": 3, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_b,
     "nombre": "b) y' = 1 + (t - y)^2"},
    {"f": f_c, "t0": 1, "y0": 2, "t_final": 2, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_c,
     "nombre": "c) y' = 1 + y/t"},
    {"f": f_d, "t0": 0, "y0": 1, "t_final": 1, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_d,
     "nombre": "d) y' = cos(2t) + sin(3t)"}
]

# Resolver y graficar cada problema
for problema in problemas:
    t, y = rkf45(problema["f"], problema["t0"], problema["y0"], problema["t_final"], problema["hmax"], problema["hmin"],
                 problema["tol"])
    y_real = problema["y_real"](t)

    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox\t y_real\t\t Error")
    for ti, yi, yri in zip(t, y, y_real):
        print(f"{ti:.2f}\t {yi:.6f}\t {yri:.6f}\t {np.abs(yi - yri):.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='Aproximación RKF45')
    plt.plot(t, y_real, 'r-', label='Solución Real')
    plt.title(f"Comparación RKF45 vs Real: {problema['nombre']}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()