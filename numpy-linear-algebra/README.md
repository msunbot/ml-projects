# Linear Algebra Basics in NumPy

This project reimplements core linear algebra operations from scratch — **matrix multiplication, rank computation (via SVD), and vector/subspace projections** — using NumPy.

It also includes performance benchmarking against built-in NumPy functions and visual demos for geometric intuition.

---

## Features
- Manual implementation of matrix multiplication
- Rank calculation using singular values
- Vector and subspace projections with QR decomposition
- Benchmark timing vs. `np.matmul`
- Interactive 2D visualization in Jupyter (`demo.ipynb`)

---

## File structure
numpy-linear-algebra/
│
├─ linalg.py         # Implementation
├─ tests.py          # Basic unit tests
├─ demo.ipynb        # Notebook visualization & timing
└─ README.md         # This file

---

## ⚙️ How to run
1. Install dependencies:
   ```
   pip install numpy matplotlib
   ```

2. Run Tests: 
    ```
    python tests.py
    ```

3. Open the notebook: 
    ```
    jupyter notebook demo.ipynb
    ```