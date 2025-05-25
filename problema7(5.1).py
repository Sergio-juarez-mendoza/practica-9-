import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# f(t, y) = -y + t + 1
def f(t, y):
    return -y + t + 1

# y0(t): constante
def y0(t):
    return 1

# y1(t)
def y1(t):
    integrand = lambda tau: f(tau, y0(tau))
    return 1 + np.array([quad(integrand, 0, ti)[0] for ti in t])

# y2(t)
def y2(t):
    y1_vals = y1(t)
    integrand = lambda tau: f(tau, np.interp(tau, t, y1_vals))
    return 1 + np.array([quad(integrand, 0, ti)[0] for ti in t])

# y3(t)
def y3(t):
    y2_vals = y2(t)
    integrand = lambda tau: f(tau, np.interp(tau, t, y2_vals))
    return 1 + np.array([quad(integrand, 0, ti)[0] for ti in t])

# Solución real
def y_real(t):
    return t + np.exp(-t)

# Dominio de t
t_vals = np.linspace(0, 1, 100)

# Evaluaciones
y0_vals = np.ones_like(t_vals)
y1_vals = y1(t_vals)
y2_vals = y2(t_vals)
y3_vals = y3(t_vals)
y_real_vals = y_real(t_vals)

# Mostrar tabla de resultados en t = 1
t1 = 1.0
y_real_1 = y_real(t1)
y1_1 = y1(np.array([t1]))[0]
y2_1 = y2(np.array([t1]))[0]
y3_1 = y3(np.array([t1]))[0]

print("RESULTADOS EN t = 1")
print("-" * 40)
print(f"{'Aproximación':<15} {'Valor':>10} {'Error absoluto':>15}")
print("-" * 40)
print(f"{'y₀(t)':<15} {1:>10.6f} {abs(1 - y_real_1):>15.6f}")
print(f"{'y₁(t)':<15} {y1_1:>10.6f} {abs(y1_1 - y_real_1):>15.6f}")
print(f"{'y₂(t)':<15} {y2_1:>10.6f} {abs(y2_1 - y_real_1):>15.6f}")
print(f"{'y₃(t)':<15} {y3_1:>10.6f} {abs(y3_1 - y_real_1):>15.6f}")
print(f"{'Exacta':<15} {y_real_1:>10.6f} {'-'*15}")

# Gráfica
plt.plot(t_vals, y0_vals, label='y₀(t)', linestyle='--')
plt.plot(t_vals, y1_vals, label='y₁(t)')
plt.plot(t_vals, y2_vals, label='y₂(t)')
plt.plot(t_vals, y3_vals, label='y₃(t)')
plt.plot(t_vals, y_real_vals, label='Solución exacta', color='black', linestyle='dotted')

plt.title("Aproximaciones por el Método de Picard")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
