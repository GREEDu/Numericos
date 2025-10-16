import numpy as np

dtype = np.float16
N = 100

def p_sum(power, order='big_to_small'):
    if order == 'big_to_small':
        seq = [1.0/(n**power) for n in range(1, N+1)]
    else:
        seq = [1.0/(n**power) for n in range(1, N+1)][::-1]
    total = dtype(0.0)
    for x in seq:
        total = dtype(total + dtype(x))
    return total

# Resultados en half precision
S2_big = p_sum(2, 'big_to_small')   # 1 + 1/4 + 1/9 + ... + 1/10^4
S2_small = p_sum(2, 'small_to_big') # 1/10^4 + ... + 1/9 + 1/4 + 1

S3_big = p_sum(3, 'big_to_small')   # 1 + 1/8 + 1/27 + ... + 1/10^6
S3_small = p_sum(3, 'small_to_big') # 1/10^6 + ... + 1/27 + 1/8 + 1

# Referencia en doble precisión
S2_ref = sum(1.0/(n**2) for n in range(1, N+1))
S3_ref = sum(1.0/(n**3) for n in range(1, N+1))

print('---- Tarea 2.4 (corregida: p=2 y p=3) ----')
print(f'sum 1/n^2 half (grande→chico) = {S2_big}')
print(f'sum 1/n^2 half (chico→grande) = {S2_small}')
print(f'sum 1/n^2 referencia (double) = {S2_ref:.6f}')
print()
print(f'sum 1/n^3 half (grande→chico) = {S3_big}')
print(f'sum 1/n^3 half (chico→grande) = {S3_small}')
print(f'sum 1/n^3 referencia (double) = {S3_ref:.6f}')

# Nota: 5.0
# Faltó analizar por qué uno de los dos métodos es más preciso que el otro.
