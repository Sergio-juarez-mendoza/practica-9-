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


# Definiciones para el Problema 1
def f_1a(t, y):
    return y * (t - (y / t) ** 2) if t != 0 else 0


def y_real_1a(t):
    return 0.5 * np.exp(t) - (1 / 3) * np.exp(t) + (1 / 3) * np.exp(-2 * t)


def f_1b(t, y):
    return 1 + (t - y) ** 2


def y_real_1b(t):
    return t + 1 / (1 - t)


def f_1c(t, y):
    return 1 + y / t if t != 0 else 0


def y_real_1c(t):
    return t * np.log(t) + 2 * t


def f_1d(t, y):
    return np.cos(2 * t) + np.sin(3 * t)


def y_real_1d(t):
    return 0.5 * np.sin(2 * t) - (1 / 3) * np.cos(3 * t) + 4 / 3


problemas_1 = [
    {"f": f_1a, "t0": 0, "y0": 0, "t_final": 1, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_1a,
     "nombre": "1a) y' = y(t - (y/t)^2)"},
    {"f": f_1b, "t0": 2, "y0": 1, "t_final": 3, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_1b,
     "nombre": "1b) y' = 1 + (t - y)^2"},
    {"f": f_1c, "t0": 1, "y0": 2, "t_final": 2, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_1c,
     "nombre": "1c) y' = 1 + y/t"},
    {"f": f_1d, "t0": 0, "y0": 1, "t_final": 1, "hmax": 0.25, "hmin": 0.05, "tol": 1e-4, "y_real": y_real_1d,
     "nombre": "1d) y' = cos(2t) + sin(3t)"}
]

for problema in problemas_1:
    t, y = rkf45(problema["f"], problema["t0"], problema["y0"], problema["t_final"], problema["hmax"], problema["hmin"],
                 problema["tol"])
    y_real = problema["y_real"](t)

    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox\t y_real\t\t Error")
    for ti, yi, yri in zip(t, y, y_real):
        print(f"{ti:.2f}\t {yi:.6f}\t {yri:.6f}\t {np.abs(yi - yri):.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'bo-', label='RKF45')
    plt.plot(t, y_real, 'r-', label='Solución Real')
    plt.title(problema["nombre"])
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()