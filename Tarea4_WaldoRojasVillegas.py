

import numpy as np
import matplotlib.pyplot as plt


def simulate_random_walk(p: float, N: int, trials: int = 10_000) -> np.ndarray:
    """Simula un paseo aleatorio unidimensional.

    Parámetros
    ----------
    p : float
        Probabilidad de dar un paso hacia la izquierda (``-1``).
    N : int
        Número de pasos en cada paseo.
    trials : int, opcional
        Número de trayectorias independientes que se simularán.

    Devuelve
    -------
    positions : ndarray de shape (trials,)
        Posición final del caminante en cada trayectoria.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p debe estar entre 0 y 1")
    q = 1.0 - p
    # Generar matrices de pasos: -1 con probabilidad p, +1 con probabilidad q
    steps = np.random.choice([-1, 1], size=(trials, N), p=[p, q])
    positions = steps.sum(axis=1)
    return positions


def main():
    # Parámetros de simulación
    N = 1000
    trials = 10_000
    p_values = [0.5, 0.3, 0.7]

    print(f"Simulación de {trials} paseos de N={N} pasos cada uno")
    for p in p_values:
        q = 1.0 - p
        positions = simulate_random_walk(p, N, trials)
        mean_sim = positions.mean()
        std_sim = positions.std()
        mean_theory = N * (q - p)
        std_theory = 2.0 * np.sqrt(p * q * N)
        # Falta deducir estas fórmulas teóricas
        print(f"\nProbabilidad de ir a la izquierda p={p} (derecha q={q})")
        print(f"Media simulada:      {mean_sim:.2f} m")
        print(f"Media teórica:       {mean_theory:.2f} m")
        print(f"Desviación std sim.: {std_sim:.2f} m")
        print(f"Desviación std teor.: {std_theory:.2f} m")

        # Histograma (opcional) – se comenta para evitar figuras en uso normal.
        plt.figure(figsize=(6, 4))
        plt.hist(positions, bins=50, density=True, alpha=0.7, color="C0")
        # Dibujar gaussiana teórica para comparar
        x = np.linspace(mean_theory - 5 * std_theory, mean_theory + 5 * std_theory, 200)
        y = (1.0 / (std_theory * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_theory) / std_theory) ** 2)
        plt.plot(x, y, 'r--', label='Gaussiana teórica')
        plt.title(f"Histograma de posiciones (p={p})")
        plt.xlabel("Posición final (m)")
        plt.ylabel("Densidad de probabilidad")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"random_walk_hist_p{p}.png")
        plt.close()

    # Respuesta corta y directa a la pregunta de la Tarea 4.8
    # La distancia típica corresponde a la desviación estándar de la posición final
    # cuando el paseo no tiene sesgo (p=0.5).  Para pasos de longitud 1 m,
    # dicha desviación estándar es la raíz cuadrada del número de pasos.
    typical_distance = np.sqrt(N)
    print(f"\nRespuesta a la Tarea 4.8: en un paseo sin sesgo (p=0.5) y pasos de 1 m,\n"
          f"la distancia típica después de {N} pasos es aproximadamente {typical_distance:.2f} m.")

# Nota: 5.5
# Faltó detallar la deducción teórica, que se podía hacer usando el teorema central del límite.
if __name__ == "__main__":
    main()
