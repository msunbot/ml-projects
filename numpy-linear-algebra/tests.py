import numpy as np
from linalg import matmul, rank, proj_vec, proj_subspace

def almost(a,b, eps=1e-9): return np.allclose(a,b,atol=eps, rtol=0)

def test_matmul():
    A = np.array([[1,2],[3,4]])
    B = np.array([[5,6],[7,8]])
    assert almost(matmul(A,B), A@B)

def test_rank():
    A = np.array([[1,2,3],[2,4,6],[1,1,1]], dtype=float)
    assert rank(A) == np.linalg.matrix_rank(A)

def test_proj_vec():
    u = np.array([3,4]); v = np.array([1,0])
    p = proj_vec(u,v)
    assert almost(p, np.array([3,0]))

def test_proj_subspace():
    u = np.array([1,2,3])
    X = np.array([[1,0,0],[0,1,0]]).T # span of e1, e2
    p = proj_subspace(u, X)
    assert almost(p, np.array([1,2,0]))

if __name__ == "__main__":
    test_matmul(); test_rank(); test_proj_vec(); test_proj_subspace()
    print("All tests passed!")
