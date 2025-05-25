import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Definiciones de los problemas
def problema_a(t, y):
    return y * np.cos(t)


def problema_b(t, y):
    return 2 / t * y + t ** 2 * np.exp(t)


def problema_c(t, y):
    return -2 / t * y + t ** 2 * np.exp(t)


def problema_d(t, y):
    return 4 * t ** 3 * y / (1 + t ** 4)


# Lista de problemas con sus condiciones
problemas = [
    {"func": problema_a, "intervalo": (0, 1), "y0": [1], "nombre": "Problema a"},
    {"func": problema_b, "intervalo": (1, 2), "y0": [0], "nombre": "Problema b"},
    {"func": problema_c, "intervalo": (1, 2), "y0": [np.sqrt(2 * np.e)], "nombre": "Problema c"},
    {"func": problema_d, "intervalo": (0, 1), "y0": [1], "nombre": "Problema d"}
]

# Tabla de resultados
print("RESULTADOS NUMÉRICOS:")
print("-" * 50)
print(f"{'Problema':<12} {'y(a)':>8} {'y(b)':>15} {'Intervalo':>12}")
print("-" * 50)

# Resolver y graficar
for p in problemas:
    a, b = p["intervalo"]
    solucion = solve_ivp(p["func"], (a, b), p["y0"], dense_output=True)

    t_vals = np.linspace(a, b, 100)
    y_vals = solucion.sol(t_vals)[0]

    # Mostrar resultado numérico
    print(f"{p['nombre']:<12} {p['y0'][0]:>8.4f} {y_vals[-1]:>15.6f} [{a}, {b}]")

    # Graficar
    plt.plot(t_vals, y_vals, label=p["nombre"])

# Gráfico
plt.title("Soluciones de los Problemas de Valor Inicial")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
