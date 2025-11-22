import numpy as np
import matplotlib.pyplot as plt


def taylor2(f, df, a, b, yo, N):

    # --- Este bloque es el usado en la clase -
    h = (b - a) / N
    t = a
    w = yo
    ts = np.zeros(N + 1)
    sol = np.zeros(N + 1)

    for i in range(N):
        ts[i] = t
        sol[i] = w
        w = w + h * f(t, w) + h * h * df(t, w) / 2
        t = t + h

    ts[N] = t
    sol[N] = w
    # --- Fin del bloque original  -

    # Derivadas en cada punto: y'(t_i) = f(t_i, y(t_i))
    daprox = np.array([f(ts[i], sol[i]) for i in range(len(ts))])

    def interp_hermite(x, puntosx, puntosy, puntosd):
        """Interpolación cúbica de Hermite en malla uniforme,
        aquí nos aseguramos de que el
        índice i no se salga de rango.
        """
        h_loc = puntosx[1] - puntosx[0]
        i = int((x - puntosx[0]) / h_loc)

        # i  en el rango válido [0, N-1]
        if i < 0:
            i = 0
        if i >= len(puntosx) - 1:
            i = len(puntosx) - 2

        f1 = (puntosy[i + 1] - puntosy[i]) / h_loc
        f21 = (f1 - puntosd[i]) / h_loc
        f22 = (puntosd[i + 1] - f1) / h_loc
        f3 = (f22 - f21) / h_loc
        dx = (x - puntosx[i])
        dx1 = (x - puntosx[i + 1])

        return puntosy[i] + puntosd[i] * dx + f21 * dx * dx + f3 * dx * dx * dx1

    def solucion_aproximada(t_eval):
        """Devuelve la aproximación y(t_eval) usando Hermite + Taylor2do."""
        return interp_hermite(t_eval, ts, sol, daprox)

    return solucion_aproximada


# ----------------------------------------------------------------------
# Ejemplo de prueba (el mismo de la clase)
# ----------------------------------------------------------------------
def _f(t, y):
    return y - t ** 2 + 1


def _df(t, y):
    # Misma expresión: df = f(t,y) - 2*t
    return _f(t, y) - 2 * t


def _exact_solution(t):
    return (t + 1) ** 2 - 0.5 * np.exp(t)


if __name__ == "__main__":
    a, b = 0.0, 2.0
    yo = 0.5
    N = 10

    aprox = taylor2(_f, _df, a, b, yo, N)

    ts = np.linspace(a, b, 200)
    ys_exacta = _exact_solution(ts)
    ys_aprox = np.array([aprox(t) for t in ts])

    plt.plot(ts, ys_exacta, label="Solución exacta")
    plt.plot(ts, ys_aprox, "--", label="Taylor 2do + Hermite (función)")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Tarea 8.6 – Método de Taylor 2do orden con interpolación de Hermite")
    plt.grid(True)
    plt.show()
