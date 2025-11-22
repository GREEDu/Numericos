import numpy as np


def rk2_step(f, t, w, h):
    """Un paso de Runge-Kutta de orden 2 (método del punto medio)."""
    k1 = f(t, w)
    k2 = f(t + h/2, w + h*k1/2)
    return h * k2


def ab2(f, a, b, yo, N):
    """
    Método de Adams-Bashforth de dos pasos para

        y'(t) = f(t, y),   a <= t <= b,   y(a) = yo.

    Usa Runge-Kutta de orden 2 para generar el primer valor y luego
    aplica Adams-Bashforth de dos pasos en el resto del intervalo.
    """
    h = (b - a) / N
    ts = np.zeros(N + 1)
    ws = np.zeros(N + 1)

    # Condición inicial
    t = a
    w = yo
    ts[0] = t
    ws[0] = w

    # Primer paso con RK2 (para ser consistente con el orden)
    k1 = f(t, w)
    k2 = f(t + h/2, w + h*k1/2)
    w1 = w + h*k2
    t1 = t + h

    ts[1] = t1
    ws[1] = w1

    # Valores de f en los dos últimos puntos
    fim1 = k1          # f(t_{0}, w_{0})
    fi = f(t1, w1)     # f(t_{1}, w_{1})

    # Pasos de Adams-Bashforth de dos pasos
    for i in range(1, N):
        t = ts[i]
        w = ws[i]
        ts[i + 1] = t + h
        # Fórmula AB2: w_{i+1} = w_i + h*(3/2 f_i - 1/2 f_{i-1})
        ws[i + 1] = w + h * (3*fi - fim1) / 2

        # Actualizamos f_{i-1} y f_i
        fim1 = fi
        fi = f(ts[i + 1], ws[i + 1])

    return ts, ws


# ----------------------------------------------------------------------
# Ejemplo 
# ----------------------------------------------------------------------
def f_tarea(t, y):
    """f(t, y) = 1 + y/t + (y/t)^2, para t >= 1."""
    return 1.0 + (y / t) + (y / t) ** 2


if __name__ == "__main__":
    a = 1.0
    b = 1.5
    yo = 0.0

    # N puede cambiar dependiendo si quieres más/menos puntos
    N = 10

    ts, ys = ab2(f_tarea, a, b, yo, N)

    # Imprimir tabla t, y(t)
    print("t\t\ty_aprox")
    for t, y in zip(ts, ys):
        print(f"{t:.4f}\t{y:.10f}")
