import numpy as np
import matplotlib.pyplot as plt


def rkf45(f, t0, y0, t_final, hmax, hmin, tol):
    t_valores = [t0]
    y_valores = [y0]
    h = hmax
    t = t0
    y = y0

    while t < t_final:
        if t + h > t_final:
            h = t_final - t

        k1 = h * f(t, y)
        k2 = h * f(t + h / 4, y + k1 / 4)
        k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = h * f(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
        k5 = h * f(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
        k6 = h * f(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)

        y4 = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
        y5 = y + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55

        error = np.abs(y5 - y4)

        if error <= tol:
            t += h
            y = y4
            t_valores.append(t)
            y_valores.append(y)

        if error != 0:
            h = min(hmax, max(hmin, 0.84 * h * (tol / error) ** 0.25))
        else:
            h = hmax

    return np.array(t_valores), np.array(y_valores)


# Definiciones para el Problema 2 (sin soluciones reales)
def f_2a(t, y):
    return (y / t) ** 2 + y / t if t != 0 else 0


def f_2b(t, y):
    return 1 / np.cos(t) + np.exp(t)


def f_2c(t, y):
    return 1 / (t ** 2 + y ** 2) if (t ** 2 + y ** 2) != 0 else 0


def f_2d(t, y):
    return t ** 2


problemas_2 = [
    {"f": f_2a, "t0": 1, "y0": 1, "t_final": 1.2, "hmax": 0.005, "hmin": 0.02, "tol": 1e-4,
     "nombre": "2a) y' = (y/t)^2 + y/t"},
    {"f": f_2b, "t0": 0, "y0": 0, "t_final": 1, "hmax": 0.25, "hmin": 0.02, "tol": 1e-4,
     "nombre": "2b) y' = sec(t) + e^t"},
    {"f": f_2c, "t0": 1, "y0": -2, "t_final": 3.2, "hmax": 0.5, "hmin": 0.02, "tol": 1e-4,
     "nombre": "2c) y' = 1/(t^2 + y^2)"},
    {"f": f_2d, "t0": 0, "y0": 0, "t_final": 2, "hmax": 0.5, "hmin": 0.02, "tol": 1e-4, "nombre": "2d) y' = t^2"}
]

for problema in problemas_2:
    t, y = rkf45(problema["f"], problema["t0"], problema["y0"], problema["t_final"], problema["hmax"], problema["hmin"],
                 problema["tol"])

    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox")
    for ti, yi in zip(t, y):
        print(f"{ti:.2f}\t {yi:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='RKF45')
    plt.title(problema["nombre"])
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()