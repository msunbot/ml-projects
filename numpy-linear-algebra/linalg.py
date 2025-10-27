import numpy as np
import time, numpy as np
from typing import Sequence, Tuple

# implement matmul 
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray: 
    A = np.asarray(A); B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2: 
        raise ValueError("A and B must be 2D") # Validate dims and raise ValueError if A.shape[1] != B.shape[0]
    m, k1 = A.shape
    k2, n = B.shape
    if k1 != k2: 
        raise ValueError(f"Incompatiable shapes {A.shape} and {B.shape}")
    C = np.zeros((m,n), dtype=np.result_type(A,B,float(64)))
    # triple loop (clear & correct); can optimize later
    for i in range(m): 
        Ai = A[i,:]
        for j in range(n):
            C[i,j] = np.dot(Ai, B[:, j])
    return C

# implement rank 
def rank(A: np.ndarray, tol: float | None = None) -> int: 
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    # Use SVD (np.linalg.svd(A, full_matrices=False)), count singular values > tol.
    if tol is None:
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        # 	If tol is None, set tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]
    return int(np.sum(s>tol))

# projections onto a vector 
def proj_vec(u: np.adarray, v:np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    denom = np.dot(v,v)
    if denom == 0:
        raise ValueError("v must be non-zero")
    return (np.dot(u,v) / denom) * v

# projections onto a subspace (columns of matrix Q)
def proj_subspace(u: np.ndarray, X:np.ndarray) -> np.ndarray:
    """Project vector u onto the column space of X."""
    u = np.asarray(u, dtype=float).reshape(-1,1)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D with basis vecotrs as columns / rows")
    Q, _ = np.linalg.qr(X)
    return (Q @ Q.T @u).reshape(-1)

# use a safer benchmarking helper (small sizes, float32, warm-up, repeats, and progress)
def benchmark_matmul(
        sizes: Sequence[Tuple[int,int]]= ((16,16),(32,32),(64,64),(96,96),(128,128)),
        repeats: int = 3,
        seed: int = 0,
        dypte=np.float32,
):
    rng = np.random.default_rng(seed) # random number generator
    rows, custom_ms, numpy_ms = [], [], []

    # warm-up (avoid first call overhead bias)
    A = rng.standard_normal((16,16), dtype=dtype)
    B = rng.standard_normal((16,16), dtype=dtype)
    _ = A@B 

    for (m,n) in sizes:
        A = rng.standard_normal((m,n), dtype=dtype)
        B = rng.standard_normal((n,m), dtype=dtype)

        # --- custom
        tmin = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = matmul(A,B) #your implmentation
            tmin = min(tmin, (time.perf_counter() - t0)*1000)
        custom_ms.append(tmin)

        # --- numpy
        tmin = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = A@B
            tmin = min(tmin, (time.perf_counter() - t0)*1000)
        numpy_ms.append(tmin)

        row.append(m) # square-ish sizes, plot vs. m
    
    return rows, custom_ms, numpy_ms