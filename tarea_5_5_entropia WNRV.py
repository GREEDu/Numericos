

import math
from math import comb, log, pi, e

def binomial_entropy(n: int, p: float) -> float:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")
    # Trivial casos de p=0 o p=1: la variable toma un único valor
    if p == 0.0 or p == 1.0:
        return 0.0
    H = 0.0
    for k in range(n + 1):
        # calcular la probabilidad exacta P(k)
        pk = comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        if pk > 0.0:
            H -= pk * math.log(pk)
    return H


def approx_binomial_entropy(n: int, p: float) -> float:
    """Calcula la aproximación asintótica de la entropía de B(n,p).
    Args:
        n: número de ensayos.
        p: probabilidad de éxito.

    Returns:
        Aproximación de la entropía en nats.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")
    if p == 0.0 or p == 1.0:
        return 0.0
    sigma2 = n * p * (1.0 - p)
    # 0.5 * ln(2 * pi * e * sigma2)
    return 0.5 * math.log(2.0 * pi * e * sigma2)


if __name__ == "__main__":
    # Ejemplo: comparar entropía exacta y aproximación para varios n
    p = 0.5
    ns = [10, 50, 100, 200]
    print(f"p = {p}\n")
    for n in ns:
        exact = binomial_entropy(n, p)
        approx = approx_binomial_entropy(n, p)
        diff = exact - approx
        print(f"n = {n:3d}: H_exact = {exact:.10f} nats, "
              f"H_approx = {approx:.10f} nats, diff = {diff:.3e}")
"""Al ejecutar este archivo como script se imprimen algunos ejemplos
comparando ambos valores para distintos tamaños ``n`` y un valor
fijo de ``p``.

La aproximación mejora rápidamente al crecer ``n``; para ``n ≥ 50``
la diferencia entre el valor exacto y la aproximación es menor
que 10⁻⁵ nats."""

# Nota: 5.0
# Retraso en la entrega.