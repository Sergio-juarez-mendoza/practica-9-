import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la reacción
k = 6.22e-19
n1 = 2e3
n2 = 2e3
n3 = 3e3

# Ecuación diferencial: dx/dt = k * (n1 - x/2)^2 * (n2 - x/2)^2 * (n3 - 3x/4)^3
def dxdt(t, x):
    return k * (n1 - x/2)**2 * (n2 - x/2)**2 * (n3 - 3*x/4)**3

# Método de Euler
def euler_method(f, t0, x0, h, steps):
    t = np.zeros(steps + 1)
    x = np.zeros(steps + 1)
    t[0], x[0] = t0, x0
    for i in range(steps):
        x[i+1] = x[i] + h * f(t[i], x[i])
        t[i+1] = t[i] + h
    return t, x

# Configuración de la simulación
t0 = 0          # Tiempo inicial (s)
x0 = 0          # Cantidad inicial de KOH (moléculas)
t_final = 0.2   # Tiempo final (s)
h = 1e-4        # Tamaño del paso (s) (pequeño para mayor precisión)
steps = int((t_final - t0) / h)

# Solución numérica
t, x = euler_method(dxdt, t0, x0, h, steps)

# Resultado en t = 0.2 s
print(f"Cantidad de KOH formado en t = {t_final} s: {x[-1]:.2f} moléculas")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'b-', label='KOH formado')
plt.title('Formación de KOH en la reacción química')
plt.xlabel('Tiempo $t$ (s)')
plt.ylabel('Cantidad de KOH $x(t)$ (moléculas)')
plt.legend()
plt.grid()
plt.show()