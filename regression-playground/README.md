# Regression Playground (NumPy → PyTorch)

Linear regression trained with gradient descent on the **California Housing** dataset.  
Phase 1: NumPy from scratch (manual gradients).  
Phase 2: PyTorch mirror (autograd).

## How to run
```
pip install numpy matplotlib scikit-learn torch
python regression_scratch.py
python regression_pytorch.py
```

## Results - Scratch
```
Epoch   1 | train MSE=5.6321 R2=-3.208 | val MSE=4.5850 R2=-2.516
Epoch  20 | train MSE=0.7135 R2=0.467 | val MSE=0.7096 R2=0.456
Epoch  40 | train MSE=0.5974 R2=0.554 | val MSE=0.6154 R2=0.528
Epoch  60 | train MSE=0.5773 R2=0.569 | val MSE=0.5941 R2=0.544
Epoch  80 | train MSE=0.5637 R2=0.580 | val MSE=0.5785 R2=0.556
Epoch 100 | train MSE=0.5538 R2=0.587 | val MSE=0.5668 R2=0.565
Epoch 120 | train MSE=0.5465 R2=0.593 | val MSE=0.5580 R2=0.572
Epoch 140 | train MSE=0.5412 R2=0.597 | val MSE=0.5514 R2=0.577
Epoch 160 | train MSE=0.5373 R2=0.600 | val MSE=0.5463 R2=0.581
Epoch 180 | train MSE=0.5344 R2=0.602 | val MSE=0.5425 R2=0.584
Epoch 200 | train MSE=0.5322 R2=0.604 | val MSE=0.5395 R2=0.586
Saved plots to /Users/msun/Documents/ai/ml-projects/regression-playground/figs
Final Val R^2: 0.586
```
- MSE - average of squared differences between predictions and true values. Lower = better
- tells you absolute error scale: good for comparing models on the same dataset 
- 0.54 means it's off by ~ sqr(0.54) ~ 0.73 units of target variable (house price in 100,000 USD)

- R^2 (coefficient of determination) - fraction of variance in the target explained by the model. Higher = better, ideally close to 1 
- tells you relative explanatory power: good for comparing across dataset 
- 0.586 means model explains 58.4% of the variance in california housing prices, the remaining 41.4% is still unexplained noise 
- in general: R^2 > 0.5 = solid baseline; 0.6 = strong linear model; 0.3 or lower = underfitting or wrong preprocessing 

## R^2 Baselines
- Linear regression - 0.55-.65
- Ridge regression (tuned lambda) - 0.60-0.70
- Random forest / boosted tree - 0.8-0.85