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

## Results - Pytorch 
```
Epoch   1 | train MSE=5.8002 R2=-3.334 | val MSE=5.6490 R2=-3.332
Epoch  20 | train MSE=1.8930 R2=-0.414 | val MSE=1.9012 R2=-0.458
Epoch  40 | train MSE=0.7023 R2=0.475 | val MSE=0.7127 R2=0.453
Epoch  60 | train MSE=0.5276 R2=0.606 | val MSE=0.5356 R2=0.589
Epoch  80 | train MSE=0.5257 R2=0.607 | val MSE=0.5337 R2=0.591
Epoch 100 | train MSE=0.5239 R2=0.609 | val MSE=0.5306 R2=0.593
Epoch 120 | train MSE=0.5235 R2=0.609 | val MSE=0.5296 R2=0.594
Epoch 140 | train MSE=0.5235 R2=0.609 | val MSE=0.5292 R2=0.594
Epoch 160 | train MSE=0.5234 R2=0.609 | val MSE=0.5291 R2=0.594
Epoch 180 | train MSE=0.5234 R2=0.609 | val MSE=0.5290 R2=0.594
Epoch 200 | train MSE=0.5234 R2=0.609 | val MSE=0.5290 R2=0.594
Final Val R^2 (torch): 0.594
``` 
- Matches numpy run (the gap is just learning rate/optimizer hynamics)

## Added Mini batch to PyTorch 
- using the same step size (0.05), the R2 became very jumpy (and sometimes go negative)
- It's because I've kept the learning rate for full batch (0.05), whereas in mini batches, each epoch has many updates, so per epoch movement is much larger 
- Fixed with AdamW: lr: 0.005 (start), try between 0.001-0.01
- weight_decay: 1e-3 
```
Ran mini batch batch size 128 and yielded this
Epoch   1 | train MSE=0.6550 R2=0.511 | val MSE=1.7472 R2=-0.340
Epoch  20 | train MSE=0.5413 R2=0.596 | val MSE=0.6071 R2=0.534
Epoch  40 | train MSE=0.5421 R2=0.595 | val MSE=0.5460 R2=0.581
Epoch  60 | train MSE=0.7031 R2=0.475 | val MSE=2.0366 R2=-0.562
Epoch  80 | train MSE=0.5372 R2=0.599 | val MSE=0.6024 R2=0.538
Epoch 100 | train MSE=0.5412 R2=0.596 | val MSE=0.5754 R2=0.559
Epoch 120 | train MSE=0.5946 R2=0.556 | val MSE=1.0382 R2=0.204
Epoch 140 | train MSE=0.5822 R2=0.565 | val MSE=0.7913 R2=0.393
Epoch 160 | train MSE=0.5729 R2=0.572 | val MSE=0.5968 R2=0.542
Epoch 180 | train MSE=0.5627 R2=0.580 | val MSE=0.8761 R2=0.328
Epoch 200 | train MSE=0.5845 R2=0.563 | val MSE=0.8631 R2=0.338
Saved plots to /Users/msun/Documents/ai/ml-projects/regression-playground/figs
Final Val R^2 (torch): 0.338

Then batch size 256 yielded this - seems more jumpy with R2 osccilates even to negative 
Epoch   1 | train MSE=0.5315 R2=0.603 | val MSE=0.5474 R2=0.580
Epoch  20 | train MSE=0.5294 R2=0.604 | val MSE=0.5327 R2=0.591
Epoch  40 | train MSE=0.5292 R2=0.605 | val MSE=0.5378 R2=0.588
Epoch  60 | train MSE=0.5472 R2=0.591 | val MSE=0.6299 R2=0.517
Epoch  80 | train MSE=0.6040 R2=0.549 | val MSE=0.7594 R2=0.418
Epoch 100 | train MSE=0.5446 R2=0.593 | val MSE=0.5782 R2=0.557
Epoch 120 | train MSE=0.8299 R2=0.380 | val MSE=3.0465 R2=-1.336
Epoch 140 | train MSE=0.5418 R2=0.595 | val MSE=0.5516 R2=0.577
Epoch 160 | train MSE=0.5479 R2=0.591 | val MSE=0.5932 R2=0.545
Epoch 180 | train MSE=0.5363 R2=0.599 | val MSE=0.5728 R2=0.561
Epoch 200 | train MSE=0.5532 R2=0.587 | val MSE=0.5990 R2=0.541
Saved plots to /Users/msun/Documents/ai/ml-projects/regression-playground/figs
Final Val R^2 (torch): 0.541

I updated the lr to 0.005 and it's fixed 
Epoch   1 | train MSE=4.2333 R2=-2.163 | val MSE=4.1484 R2=-2.181
Epoch  20 | train MSE=0.5252 R2=0.608 | val MSE=0.5341 R2=0.590
Epoch  40 | train MSE=0.5244 R2=0.608 | val MSE=0.5302 R2=0.593
Epoch  60 | train MSE=0.5240 R2=0.608 | val MSE=0.5300 R2=0.594
Epoch  80 | train MSE=0.5247 R2=0.608 | val MSE=0.5293 R2=0.594
Epoch 100 | train MSE=0.5252 R2=0.608 | val MSE=0.5280 R2=0.595
Epoch 120 | train MSE=0.5255 R2=0.607 | val MSE=0.5344 R2=0.590
Epoch 140 | train MSE=0.5247 R2=0.608 | val MSE=0.5300 R2=0.594
Epoch 160 | train MSE=0.5242 R2=0.608 | val MSE=0.5334 R2=0.591
Epoch 180 | train MSE=0.5246 R2=0.608 | val MSE=0.5287 R2=0.595
Epoch 200 | train MSE=0.5242 R2=0.608 | val MSE=0.5305 R2=0.593
Saved plots to /Users/msun/Documents/ai/ml-projects/regression-playground/figs
Final Val R^2 (torch): 0.593
```


## Reflections
- Why PyTorch takes ~50 epochs vs. NumPy ~10
- Differences: 
1) Learning rate: NumPy uses fixed step size, PyTorch uses adaptive per-parameter rate, smaller initialy 
2) Optimizer: NumPy used vanilla Gradient Descent, PyTorch used AdamW (slower start, smooth convergence), accumulates gradient estimates before taking bigger steps 
3) Weight initialization: NumPy used Normal(0, 0.01), PyTorch started with smaller weights (used Uniform(default nn.Linear))
4) Regularization: NumPy used explicit L2 penalty, PyTorch used built-in weight_decay, different scaling constant 