import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# Tarea 10.10 - Ajuste de modelos mediante mínimos cuadrados
# ---------------------------------------------------------

# Datos del primer conjunto
x1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
y1 = np.array([0.280, 0.472, 0.686, 0.850, 1.263, 1.576, 1.894])
sigma1 = np.array([0.10, 0.07, 0.05, 0.10, 0.03, 0.01, 0.01])

# Datos del segundo conjunto
x2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
y2 = np.array([-1.36, 0.20, 3.01, 0.30, 1.11, 1.52, 2.09])
sigma2 = np.array([2.0, 1.4, 1.0, 2.0, 0.6, 0.2, 0.2])

# ------------------------
# Modelos para el ajuste:
# ------------------------

def modelo1(x, a1):
    return a1 * x

def modelo2(x, a1, a2):
    return a1 * x + a2 * x**2

def modelo3(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x**2

# ------------------------
# Ajuste general con pesos
# ------------------------

def ajustar(modelo, x, y, sigma):
    popt, pcov = curve_fit(modelo, x, y, sigma=sigma, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    y_fit = modelo(x, *popt)
    chi2 = np.sum(((y - y_fit) / sigma)**2)
    dof = len(y) - len(popt)
    chi2_red = chi2 / dof
    return popt, perr, chi2, chi2_red

# ------------------------
# Mostrar resultados
# ------------------------

def mostrar(nombre, resultados):
    print(f"\n===== {nombre} =====")
    for i, (popt, perr, chi2, chi2_red) in enumerate(resultados, start=1):
        print(f"\nModelo {i}:")
        for j, (p, e) in enumerate(zip(popt, perr)):
            print(f"  a{j} = {p:.6f} ± {e:.6f}")
        print(f"  chi^2      = {chi2:.6f}")
        print(f"  chi^2_red  = {chi2_red:.6f}")

# ------------------------
# Programa principal
# ------------------------

if __name__ == "__main__":

    # Ajustes para datos 1
    r1_m1 = ajustar(modelo1, x1, y1, sigma1)
    r1_m2 = ajustar(modelo2, x1, y1, sigma1)
    r1_m3 = ajustar(modelo3, x1, y1, sigma1)
    mostrar("Datos 1", [r1_m1, r1_m2, r1_m3])

    # Ajustes para datos 2
    r2_m1 = ajustar(modelo1, x2, y2, sigma2)
    r2_m2 = ajustar(modelo2, x2, y2, sigma2)
    r2_m3 = ajustar(modelo3, x2, y2, sigma2)
    mostrar("Datos 2", [r2_m1, r2_m2, r2_m3])

    print("\nEl mejor modelo es el que tiene chi^2 reducido más cercano a 1.")
