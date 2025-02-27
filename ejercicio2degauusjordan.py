import numpy as np

def gauss_jordan_pivot_determinante(A, b):
    n = len(A)
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    det_A = np.linalg.det(A)
    
    if np.isclose(det_A, 0):
        print(f"Determinante de A: {det_A:.5f}. El sistema es indeterminado o no tiene solución única.")
        return None
    
    print(f"Determinante de A: {det_A:.5f}. El sistema tiene solución única.")
    
    for i in range(n):
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        Ab[i] = Ab[i] / Ab[i, i]
        
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]
    
    return Ab[:, -1]

A2 = np.array([
    [3, -2, 5, -1, 4, 2, -3, 1, 2],
    [-2, 4, -3, 1, 5, -1, 2, -4, 3],
    [5, -1, 2, -3, 4, 1, -2, 3, -1],
    [1, -3, 4, -2, 5, -1, 2, -1, 4],
    [2, 3, -1, 4, -2, 5, -3, 1, -2],
    [-3, 2, 4, -1, 3, -2, 5, -1, 1],
    [4, -1, 3, 2, -3, 1, -2, 5, -4],
    [-1, 5, -2, 3, 4, -1, 2, -3, 1],
    [3, -2, 5, -1, 4, 2, -3, 1, -5]
], dtype=float)

b2 = np.array([-8, 7, -6, 5, 12, -9, 10, 3, -2], dtype=float)

sol2 = gauss_jordan_pivot_determinante(A2, b2)
print("Solución del sistema:", sol2)
