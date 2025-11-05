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

## Visualization 
- vis.ipynb provides visual comparisons of logistic regression of 2D toy data (make_moons) using 
- Numpy (from-scratch GD) - deg 1 & deg 2 feature maps
- PyTorch (mini-batch AdamW) - deg3 feature map
- PyTorch MLP on raw(x1, x2) for a nonlinear baseline

## Results 
Numpy (deg 1) val acc: 0.85; Numpy (deg 2, quad): 0.86; PyTorch Logistic (deg 3 full cubic): 0.96; PyTorch MLP (2x64 ReLU): 0.97-0.99 

## How to Run 
```
cd decision-boundary-visualizer
python3 logreg_numpy.py
# degree-2 (sanity ~0.86)
python3 logreg_torch.py --deg 2 --lr 0.01 --lam 0.00 --epochs 1500
# degree-3 (best ~0.95–0.96)
python3 logreg_torch.py --deg 3 --lr 0.01 --lam 0.03 --epochs 2000
```

## Key Learnings
- Feature maps (deg2/deg3) let a linear model learn non-linear boundaries
- Regularization (L2 / weight decay) smooths the curve; too much makes it linear again
- BCEWithLogitsLoss + AdamW is a stable PyTorch recipe for logistic regression
- An MLP on raw input often outperforms polynominal LR with less manual feature work 
