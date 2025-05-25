import numpy as np
import matplotlib.pyplot as plt


def taylor_orden_2(f, df, t0, y0, h, pasos):
    """
    Aplica el método de Taylor de orden 2 para resolver una EDO.

    Parámetros:
    - f: Función dy/dt = f(t, y)
    - df: Derivada total de f respecto a t (df/dt + df/dy * f)
    - t0: Valor inicial de t
    - y0: Valor inicial de y
    - h: Tamaño del paso
    - pasos: Número de pasos a realizar

    Retorna:
    - t_valores: Array de valores de t
    - y_valores: Array de valores aproximados de y
    """
    t_valores = np.zeros(pasos + 1)
    y_valores = np.zeros(pasos + 1)

    t_valores[0] = t0
    y_valores[0] = y0

    for i in range(pasos):
        t = t_valores[i]
        y = y_valores[i]
        # Método de Taylor orden 2: y_{n+1} = y_n + h*f(t_n, y_n) + (h²/2)*df(t_n, y_n)
        y_valores[i + 1] = y + h * f(t, y) + (h ** 2 / 2) * df(t, y)
        t_valores[i + 1] = t + h

    return t_valores, y_valores


# Definir las funciones f(t, y) y sus derivadas totales df/dt para cada inciso
# a) y' = te^(3t) - 2y
def f_a(t, y):
    return t * np.exp(3 * t) - 2 * y


def df_a(t, y):
    return np.exp(3 * t) * (3 * t + 1) - 2 * f_a(t, y)  # df/dt = e^(3t)(3t + 1) - 2dy/dt


# b) y' = 1 + (t - y)^2
def f_b(t, y):
    return 1 + (t - y) ** 2


def df_b(t, y):
    return 2 * (t - y) * (1 - f_b(t, y))  # df/dt = 2(t - y)(1 - dy/dt)


# c) y' = 1 + y/t
def f_c(t, y):
    return 1 + y / t


def df_c(t, y):
    return -y / t ** 2 + f_c(t, y) / t  # df/dt = -y/t² + (1 + y/t)/t


# d) y' = cos(2t) + sin(3t)
def f_d(t, y):
    return np.cos(2 * t) + np.sin(3 * t)


def df_d(t, y):
    return -2 * np.sin(2 * t) + 3 * np.cos(3 * t)  # df/dt no depende de y


# Configuración para cada problema
problemas = [
    {"f": f_a, "df": df_a, "t0": 0, "y0": 0, "h": 0.5, "t_final": 1, "nombre": "a) y' = te^{3t} - 2y"},
    {"f": f_b, "df": df_b, "t0": 2, "y0": 1, "h": 0.5, "t_final": 3, "nombre": "b) y' = 1 + (t - y)^2"},
    {"f": f_c, "df": df_c, "t0": 1, "y0": 2, "h": 0.25, "t_final": 2, "nombre": "c) y' = 1 + y/t"},
    {"f": f_d, "df": df_d, "t0": 0, "y0": 1, "h": 0.25, "t_final": 1, "nombre": "d) y' = cos(2t) + sin(3t)"}
]

# Resolver y mostrar resultados para cada problema
for problema in problemas:
    pasos = int((problema["t_final"] - problema["t0"]) / problema["h"])
    t, y = taylor_orden_2(problema["f"], problema["df"], problema["t0"], problema["y0"], problema["h"], pasos)

    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox")
    for ti, yi in zip(t, y):
        print(f"{ti:.2f}\t {yi:.6f}")

    # Graficar
    plt.figure()
    plt.plot(t, y, 'bo-', label="Aproximación Taylor Orden 2")
    plt.title(f"Método de Taylor Orden 2: {problema['nombre']}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()