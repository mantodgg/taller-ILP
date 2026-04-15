import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
 
# =============================================================================
# PROBLEMA 1: Maximizar z = 25x + 50y
# Restricciones:
#   x + y  ≤ 90
#   4x + 6y ≥ 390
#   15x + 40y ≤ 2000
#   x, y ≥ 0
# =============================================================================
 
c = [-25, -50]  # negativos porque milp minimiza
 
A = [
    [1,  1],
    [4,  6],
    [15, 40]
]
 
bu = [90,      np.inf, 2000]
bl = [-np.inf, 390,    -np.inf]
 
constraints = LinearConstraint(A, bl, bu)
bounds      = Bounds([0, 0], [np.inf, np.inf])
 
res = milp(
    c=c,
    constraints=constraints,
    bounds=bounds,
    integrality=[1, 1]  # 1 = entera, 0 = continua
)
 
print("=== PROBLEMA 1 ===")
print("Estado:", res.message)
print("F =",    -res.fun)
print("x =",    res.x[0])
print("y =",    res.x[1])
print()
 
# =============================================================================
# PROBLEMA 2: Maximizar z = 3x + 5y + 7z
# Restricciones:
#   x +  y +  z ≤ 12
#   x + 2y + 3z ≤ 18
#   x +  y + 2z ≤ 14
#   x, y, z ≥ 0
# =============================================================================
 
c2 = [-3, -5, -7]
 
B = [
    [1, 1, 1],
    [1, 2, 3],
    [1, 1, 2]
]
 
cot_sup2 = [12, 18, 14]
cot_inf2 = [-np.inf, -np.inf, -np.inf]
 
constraints2 = LinearConstraint(B, cot_inf2, cot_sup2)
bounds2      = Bounds([0, 0, 0], [np.inf, np.inf, np.inf])
 
res2 = milp(
    c=c2,
    constraints=constraints2,
    bounds=bounds2,
    integrality=[1, 1, 1]
)
 
print("=== PROBLEMA 2 ===")
print("Estado:", res2.message)
print("F =",    -res2.fun)
print("x =",    res2.x[0])
print("y =",    res2.x[1])
print("z =",    res2.x[2])
print()
 
# =============================================================================
# PROBLEMA 3: Asignación (15 variables binarias)
# ============================================================================= 

c3 = [0] * 15
 
C = [
    [7, 10, 6, 2, 4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [0,  0, 0, 0, 0,  7, 10,  6,  2,  4,  0,  0,  0,  0,  0],
    [0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  7, 10,  6,  2,  4],
    [1,  0, 0, 0, 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [0,  1, 0, 0, 0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0],
    [0,  0, 1, 0, 0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0],
    [0,  0, 0, 1, 0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0],
    [0,  0, 0, 0, 1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1],
]
 
cot_sup3 = [12, 12, 12, 1, 1, 1, 1, 1]
cot_inf3 = [ 8,  8,  8, 1, 1, 1, 1, 1]
 
constraints3 = LinearConstraint(C, cot_inf3, cot_sup3)
bounds3      = Bounds([0] * 15, [np.inf] * 15)
 
res3 = milp(
    c=c3,
    constraints=constraints3,
    bounds=bounds3,
    integrality=[1] * 15
)
 
print("=== PROBLEMA 3 ===")
print("Estado:", res3.message)
print("F =",    -res3.fun)
for i, val in enumerate(res3.x):
    print(f"x{i+1} =", val)
    
# =============================================================================
# PROBLEMA 4: Asignación 
# En este ejercicio debemos asignarle a cada persona un empleo distinto, para ser asignado debe haber sacado el mejor puntaje posible pero además no haber sido elegido
# en otro empleo. Es decir la idea es optimizar para que cada trabajo tenga el mejor empleado posible. 

# Para simplificar la consigna vamos a hacerlo con 4 empleos y 4 personas.
# Nuestra matriz va a quedar tal que asi:
# puntajes = [
#  [8, 3, 6, 2],  # A 
#   [4, 7, 5, 9],  # B
#   [6, 5, 8, 3],  # C
#   [2, 9, 4, 7]]  # D

# A,B,C y D van a ser los empleos y 1,2,3 y 4 las personas.
# =============================================================================   

c4 = [-8, -3, -6, -2, -4, -7, -5, -9, -6, -5, -8, -3, -2, -9, -4, -7] #funcion objetivo

col1 = [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0] #cada empleo recibe un empleado
col2 = [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0]
col3 = [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0]
col4 = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]

filA = [1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0] #cada empleado recibe un empleo
filB = [0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0]
filC = [0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0]
filD = [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1]

D = [col1, col2, col3, col4, filA, filB, filC, filD]

cot_sup4 = [1]*8
cot_inf4 = [1]*8

constraints4 = LinearConstraint(D, cot_inf4, cot_sup4)
bounds4 = ([0] *16, [np.inf]*16)

res4 = milp(
    c=c4,
    constraints=constraints4,
    bounds=bounds4,
    integrality=[1]*16
)

print("=== PROBLEMA 4 ===")
print("Estado:", res4.message)
print("F =",    -res4.fun)
print("A1 =", res4.x[0])
print("A2 =", res4.x[1])
print("A3 =", res4.x[2])
print("A4 =", res4.x[3])
print("B1 =", res4.x[4])
print("B2 =", res4.x[5])
print("B3 =", res4.x[6])
print("B4 =", res4.x[7])
print("C1 =", res4.x[8])
print("C2 =", res4.x[9])
print("C3 =", res4.x[10])
print("C4 =", res4.x[11])
print("D1 =", res4.x[12])
print("D2 =", res4.x[13])
print("D3 =", res4.x[14])
print("D4 =", res4.x[15]) #








