import numpy as np

# Parámetros
q_max = 0.1
N = 500000
dq = q_max / N

# -------------------------
# Método 1: Calcular I1 e I2 por separado
# -------------------------
def integrando_I1(q):
    return q**2 * (1/q**5 + 3/q)

def integrando_I2(q):
    return q**2 * (-1/q**5 + 1/q)

I1 = 0.0
I2 = 0.0

for i in range(1, N+1):
    q = i * dq
    I1 += integrando_I1(q)
    I2 += integrando_I2(q)

I1 *= dq
I2 *= dq
I_sep = I1 + I2

# -------------------------
# Método 2: Simplificar primero (integrando combinado)
# -------------------------
def integrando_total(q):
    return 4 * q

I_total = 0.0
for i in range(1, N+1):
    q = i * dq
    I_total += integrando_total(q)

I_total *= dq

# -------------------------
# Resultado exacto
# -------------------------
I_exact = 2 * q_max**2

print("---- Tarea 2.3 ----")
print(f"I1 ≈ {I1:.6f}")
print(f"I2 ≈ {I2:.6f}")
print(f"Suma I1+I2 (separado) ≈ {I_sep:.6f}")
print(f"Integral combinada ≈ {I_total:.6f}")
print(f"Valor exacto = {I_exact:.6f}")
print("\nNota: la diferencia muestra la cancelación catastrófica al separar.")

