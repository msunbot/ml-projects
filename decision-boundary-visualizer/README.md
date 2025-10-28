# Decision Boundary Visualizer (PyTorch vs. Numpy)

This is a hands-on project to visualize how logistic regression classifies non-linear data. 
Built first in NumPy (manual gradient descent) and then in PyTorch (with mini-batch AdamW)
This project refinroces concepts from Coursera's Machine Learning Specialization -- especially classification, regulariation and feature mapping.

---

## Concept Reinforced

- Logistic Regression: Binary classification via sigmoid and cross-entropy loss
- Decision Boundary: Visualizing how models separate classes in 2D space
- Polynomial Feature Expansion: Turning linear models into non-linear classifiers
- Regularization: L2 penalty (weight decay) to smooth boundaries & prevent overfitting 
- PyTorch 

---

## Files

``` 
decision-boundary-visualizer/
├─ logreg_numpy.py          # NumPy implementation (manual gradient descent)
├─ logreg_torch.py          # PyTorch implementation (mini-batch AdamW)
├─ figs/
│   ├─ logreg_scratch_linear.png   # Linear boundary (deg=1)
│   ├─ logreg_scratch_poly3.png    # NumPy polynomial boundary
│   └─ logreg_torch_boundary.png   # PyTorch polynomial boundary (deg=3)
└─ README.md
```

## Implementation 

### NumPy version (logreg_numpy.py)
- Implements logistic regression from scratch
- Gradient descent with L2 regularization 
- Polynomial feature mapping (d2 and 3)
- Trains on make_moons dataset and plots decision boundary 

### PyTorch version (logreg_torch.py)
- Reimplement the same model using PyTorch
- Uses BCEWithLogitsLoss (numerically stable) + AdamW optimizer
- Supports mini-batch gradient descent, early stopping, best checkpoint saving
- Feature expansion handled in NumPy, then fed to PyTorch tensors

---

## Experiments & Results
1. Linear: .85
2. Quadratic: .86 - adds mild curvature 
3. Cubic: .96 - capture moon shape accurately

---

## Key Learnings
- Linear logistic regression only draws a single straight boundary
- Polynomial features transform the input space so logistic regression can learn non-linear separation
- Regularization (L2) smooths the boundary and prevents overfitting to noisy samples
- In PyTorch implementation, BCEWithLogitsLoss + AdamW replicates classical logistic regression while being stable and scalable
- Plotting decision regions helps visualizing how feature transformations affect separability. 

---

## How to Run 
1. NumPy version 
``` 
cd decision-boundary-visualizer
python logreg_numpy.py 
``` 

2. PyTorch version
```
# Degree 2 (mild curvature)
python logreg_torch.py --deg 2 --lr 0.01 --lam 0.01 --epochs 1500

# Degree 3 (best-performing)
python logreg_torch.py --deg 3 --lr 0.01 --lam 0.03 --epochs 2000
```
Outputs are saved in the figs/ folder 
---

