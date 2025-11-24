#%%

import numpy as np

# Excepción igual que en las deñ curso
class MaxIterations(Exception):
    pass

# ------------------------------
# Regla compuesta del trapecio
# ------------------------------
def trapecio(f, a, b, n):
    """
    Aproxima ∫_a^b f(x) dx usando la regla compuesta del trapecio
    con n subintervalos.
    """
    h = (b - a) / n
    # puntos interiores
    x = np.linspace(a + h, b - h, n - 1, endpoint=True)
    return (h / 2) * (f(a) + 2 * np.sum(f(x)) + f(b))

# ------------------------------
# Densidad de la normal estándar
# φ(t) = 1/sqrt(2π) * exp(-t^2/2)
# ------------------------------
def normal_pdf(t):
    return np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)

# Número de subintervalos para la integral
N_TRAP = 1000

# ------------------------------
# F(p) = ∫_0^p φ(t) dt - 0.45
# ------------------------------
def F(p):
    return trapecio(normal_pdf, 0.0, p, N_TRAP) - 0.45

# F'(p) = φ(p)
def dF(p):
    return normal_pdf(p)

# ------------------------------
# Método de Newton
# ------------------------------
def newton(f, df, po, epsilon, N=1_000_000):
    """
    Método de Newton para f(x)=0.
    Convergencia cuando |p_n - p_{n-1}|/|p_n| < epsilon.
    """
    for i in range(N):
        p = po - f(po) / df(po)
        if abs(p - po) / abs(p) < epsilon:
            return p, i + 1
        po = p

    raise MaxIterations("No se encontró el cero luego de {} iteraciones".format(N))

# ------------------------------
# Programa principal
# ------------------------------
if __name__ == "__main__":
    # Punto inicial
    p0 = 1.5
    tol = 1e-5

    raiz, iteraciones = newton(F, dF, p0, tol)

    print(f"x ≈ {raiz:.10f}")
    print(f"Iteraciones de Newton: {iteraciones}")

    # Verificación de la integral
    valor_integral = trapecio(normal_pdf, 0.0, raiz, N_TRAP)
    print(f"∫0^x φ(t) dt ≈ {valor_integral:.8f}")

# Nota: 5.0
# Retraso en la entrega.
# Demasiado uso de IA sin citarla adecuadamente.
