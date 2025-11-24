# %%

import numpy as np

try:
    import matplotlib.pyplot as plt  # para graficar si está disponible
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


def compute_interpolation(n_points: int = 10, a: float = -10.0, b: float = 10.0):
    # Genera puntos equidistantes
    x_nodes = np.linspace(a, b, n_points)
    # Evalua la funcion f(x) = tanh(x)
    y_nodes = np.tanh(x_nodes)
    # Computo polinomial
    degree = n_points - 1
    coeffs = np.polyfit(x_nodes, y_nodes, degree)
    return coeffs, x_nodes, y_nodes


def evaluate_polynomial(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(coeffs, x)


def main():
    # computo interpolacion
    coeffs, x_nodes, y_nodes = compute_interpolation(n_points=10, a=-10.0, b=10.0)
    # evaluacion
    x_grid = np.linspace(-10.0, 10.0, 1000)
    f_vals = np.tanh(x_grid)
    p_vals = evaluate_polynomial(coeffs, x_grid)
    # Error computo
    abs_error = np.abs(p_vals - f_vals)
    # Print summary
    print("Resumen de la interpolación de tanh(x) con 10 puntos equidistantes entre -10 y 10")
    print("====================================================================")
    print(f"Número de puntos de interpolación: {len(x_nodes)} (grado del polinomio = {len(x_nodes)-1})")
    print(f"Máximo error absoluto en la malla de 1000 puntos: {np.max(abs_error):.3e}")
    # También mostrar error en extremos y en la región central
    mid_indices = (x_grid > -1) & (x_grid < 1)
    max_error_mid = np.max(abs_error[mid_indices])
    print(f"Máximo error absoluto en el intervlo [-1, 1] (región central): {max_error_mid:.3e}")
    end_indices = (x_grid < -8) | (x_grid > 8)
    max_error_ends = np.max(abs_error[end_indices])
    print(f"Máximo error absoluto en |x|>8 (cerca de los extremos): {max_error_ends:.3e}")
    # interpretacion
    print("\nInterpretación:")
    print(
        "El polinomio de grado 9 interpola exactamente la función tanh(x) en diez "
        "puntos equidistantes. Sin embargo, la función tanh(x) tiene saturación "
        "(tiende a ±1) para |x| grandes, mientras que un polinomio de alto grado no puede "
        "replicar este comportamiento asintótico. Por ello, se observan grandes oscilaciones "
        "y errores en los extremos del intervalo. En la región central "
        "cercana a x=0, la aproximación es mucho mejor."
    )
    # plotear
    if HAVE_PLT:
        plt.figure(figsize=(8, 4))
        plt.plot(x_grid, f_vals, label="tanh(x)")
        plt.plot(x_grid, p_vals, label="Polinomio interpolante (grado 9)")
        plt.scatter(x_nodes, y_nodes, color="red", zorder=5, label="Puntos de interpolación")
        plt.title("Interpolación de tanh(x) usando 10 puntos equidistantes")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

# Nota: 5.0
# Retraso en la entrega.
