import numpy as np
import matplotlib.pyplot as plt


def metodo_euler(f, t0, y0, h, pasos):
    """
    Aproxima la solución de una EDO usando el método de Euler.

    Parámetros:
    - f: Función dy/dt = f(t, y)
    - t0: Valor inicial de t
    - y0: Valor inicial de y
    - h: Tamaño del paso
    - pasos: Número de pasos

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
        y_valores[i + 1] = y + h * f(t, y)
        t_valores[i + 1] = t + h

    return t_valores, y_valores


# Definición de las EDOs para cada inciso
def f_a(t, y):
    return t * np.exp(3 * t) - 2 * y  # y' = te^(3t) - 2y


def f_b(t, y):
    return 1 + (t - y) ** 2  # y' = 1 + (t - y)^2


def f_c(t, y):
    return 1 + y / t  # y' = 1 + y/t (asumo que "yh" es un typo y debería ser y/t)


def f_d(t, y):
    return np.cos(2 * t) + np.sin(3 * t)  # y' = cos(2t) + sin(3t)


# Configuración para cada problema
problemas = [
    {"f": f_a, "t0": 0, "y0": 0, "h": 0.5, "t_final": 1, "nombre": "a) y' = te^{3t} - 2y"},
    {"f": f_b, "t0": 2, "y0": 1, "h": 0.5, "t_final": 3, "nombre": "b) y' = 1 + (t - y)^2"},
    {"f": f_c, "t0": 1, "y0": 2, "h": 0.25, "t_final": 2, "nombre": "c) y' = 1 + y/t"},
    {"f": f_d, "t0": 0, "y0": 1, "h": 0.25, "t_final": 1, "nombre": "d) y' = cos(2t) + sin(3t)"}
]

# Resolver y mostrar resultados para cada problema
for problema in problemas:
    pasos = int((problema["t_final"] - problema["t0"]) / problema["h"])
    t, y = metodo_euler(problema["f"], problema["t0"], problema["y0"], problema["h"], pasos)

    print(f"\n--- {problema['nombre']} ---")
    print("t\t y_aprox")
    for ti, yi in zip(t, y):
        print(f"{ti:.2f}\t {yi:.6f}")

    # Graficar
    plt.figure()
    plt.plot(t, y, 'bo-', label="Aproximación Euler")
    plt.title(f"Método de Euler: {problema['nombre']}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()