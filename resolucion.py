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