import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, Eq, Derivative, dsolve, exp

# -------------------------------
# 1. Demostración Analítica (Inciso a)
# -------------------------------
print("\n--- Parte a: Derivación de la ecuación para p(t) ---")
t, b, d, r = symbols('t b d r')
x = Function('x')(t)
xn = Function('x_n')(t)
p = Function('p')(t)

# Ecuaciones originales
dx_dt = Eq(Derivative(x, t), (b - d) * x)
dxn_dt = Eq(Derivative(xn, t), (b - d) * xn + r * b * (x - xn))

# Definición de p(t) = xn(t)/x(t)
p_definition = Eq(p, xn / x)

# Derivamos p(t) usando regla del cociente
dp_dt = Derivative(p, t).doit().subs({
    Derivative(xn, t): dxn_dt.rhs,
    Derivative(x, t): dx_dt.rhs
}).simplify()

print(f"Ecuación simplificada para dp/dt: {dp_dt} = {dp_dt.simplify()}")

# -------------------------------
# 2. Solución Numérica con Euler (Inciso b)
# -------------------------------
print("\n--- Parte b: Aproximación Numérica con Método de Euler ---")

# Parámetros
b_val = 0.02
d_val = 0.015
r_val = 0.1
p0 = 0.01
t_final = 50
h = 1  # Tamaño de paso (1 año)

# Definición de la EDO dp/dt = r*b*(1 - p)
def dpdt(t, p):
    return r_val * b_val * (1 - p)

# Método de Euler
def euler_method(f, t0, y0, h, steps):
    t = np.zeros(steps + 1)
    y = np.zeros(steps + 1)
    t[0], y[0] = t0, y0
    for i in range(steps):
        y[i+1] = y[i] + h * f(t[i], y[i])
        t[i+1] = t[i] + h
    return t, y

# Calculamos pasos
steps = int(t_final / h)
t_num, p_num = euler_method(dpdt, 0, p0, h, steps)

# Resultado en t=50
print(f"Aproximación numérica en t=50: p(50) ≈ {p_num[-1]:.6f}")

# -------------------------------
# 3. Solución Exacta (Inciso c)
# -------------------------------
print("\n--- Parte c: Solución Exacta y Comparación ---")

# Resolución simbólica con sympy
p_exact_eq = dsolve(Eq(Derivative(p, t), r * b * (1 - p)), p, ics={p.subs(t, 0): p0})
p_exact = p_exact_eq.rhs.subs({b: b_val, r: r_val})

# Evaluamos en t=50
p_exact_50 = p_exact_eq.rhs.subs({t: 50, b: b_val, r: r_val}).evalf()
print(f"Solución exacta en t=50: p(50) = {p_exact_50:.6f}")

# Error absoluto
error = abs(p_num[-1] - float(p_exact_50))
print(f"Error absoluto: {error:.10f}")

# -------------------------------
# 4. Gráficas
# -------------------------------
# Puntos para la solución exacta
t_exact_vals = np.linspace(0, t_final, 500)
p_exact_vals = [1 - (1 - p0) * np.exp(-r_val * b_val * ti) for ti in t_exact_vals]

# Configuración de la gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_num, p_num, 'bo-', label=f'Aproximación Euler (h={h})', markersize=4)
plt.plot(t_exact_vals, p_exact_vals, 'r-', label='Solución Exacta')
plt.title('Evolución de la Proporción de No Conformistas $p(t)$')
plt.xlabel('Tiempo $t$ (años)')
plt.ylabel('Proporción $p(t)$')
plt.legend()
plt.grid()
plt.show()