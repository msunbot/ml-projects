# MNIST Digit Classifier (PyTorch)

This project trains a simple neural network to recognize handwritten digits using the **MNIST dataset**, implemented from scratch in PyTorch.

It includes two models: 
1. **Softmax Regression (Logistic Regression)**
2. **Multilayer Perceptron (MLP)**

---

## Features
- Logistic regression (no hidden layer)
- Fully connected neural net (1 hidden layer + ReLU)
- Training & evaluation scripts 
- Confusion matrix and performance report

--- 
## How to run
```
pip install torch torchvision matplotlib scikit-learn tqdm
python train_logreg.py
python train_mlp.py
```
## Check accuracy and confusion matrix
```
python eval.py
```
--- 
## Results
-- Logreg 
```
Epoch 1: train acc=0.8557 test acc=0.9074                                       
Epoch 2: train acc=0.9108 test acc=0.9165                                       
Epoch 3: train acc=0.9166 test acc=0.9205                                       
Epoch 4: train acc=0.9212 test acc=0.9252                                       
Epoch 5: train acc=0.9232 test acc=0.9244                                       
Best test acc: 0.9252
``` 
Accuracy is within expected range of 92093% 

-- Multilayer perceptron 
```
Epoch 1L train 0.9024, test 0.9548                                              
Best: 0.9548
Epoch 2L train 0.9551, test 0.9647                                              
Best: 0.96475
Epoch 3L train 0.9666, test 0.9792                                              
Best: 0.9791666666666666
Epoch 4L train 0.9741, test 0.9838                                              
Best: 0.9838
Epoch 5L train 0.9778, test 0.9880                                              
Best: 0.98805
Epoch 6L train 0.9812, test 0.9890                                              
Best: 0.9890333333333333
Epoch 7L train 0.9846, test 0.9927                                              
Best: 0.9927
Epoch 8L train 0.9858, test 0.9918                                              
Best: 0.9927
```
Accuracy is 99.27%, better than expected 97-98% because
1. MNIST is relatively easy dataset, clean, centered, grayscale. 
1 hidden layer MLP with 256 neurons has ~200k params, plenty to model the training set perfectly 
With good initialization and AdamW, it can reach 99% easily.

Log Reg = 92-93% 
1-hidden-layer MLP = 97-99%
Simple CNN = 99.2-99.5%
Big CNN (LeNet5, ResNet) = 99.7-99.8%

2. ReLU, Dropout, AdamW are excellent optimization 
- ReLU activation: gradient never vanishes in positive region, lets training converge quickly
- Dropout(0.2): adds light regularization, prevents overfitting but still leaves full capacity
- AdamW optimizer : adaptive learning rates, decoupled weight decay, allowing stable optimization 

3. Normalization & modern defaults 
- In utils.py, normalize MNIST with mean (0.1307) and std (0.3081)
- That centers the data and makes optimization smoother
- Paried with modest learning rate (1e-3) and batch size = 256, training curve stabilizes quickly 

## Results from eval.py 

```
Confusion matrix:
 [[ 973    1    1    0    0    0    2    1    2    0]
 [   0 1127    3    1    0    0    2    0    2    0]
 [   6    2 1011    0    1    0    1    6    5    0]
 [   0    0    4  988    0    3    0    6    5    4]
 [   2    0    5    0  961    0    3    2    0    9]
 [   2    0    0    6    1  876    2    1    3    1]
 [   6    3    2    1    2    4  940    0    0    0]
 [   0    5   10    3    1    0    0 1004    1    4]
 [   4    1    2    5    6    1    2    4  947    2]
 [   4    4    0    4    9    4    1    7    1  975]]
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.97      0.98      0.98      1032
           3       0.98      0.98      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.99      0.98      0.98       892
           6       0.99      0.98      0.98       958
           7       0.97      0.98      0.98      1028
           8       0.98      0.97      0.98       974
           9       0.98      0.97      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
``` 

What it represents: 
- table showing what the model predicted vs. what's actually true.
- Rows: actual label (ground truth), Columns: predicted (model guesses)
- Cell (i,j): number of times a sample of true class i was predicted as class j
- Diagonal numbers are correct predictions (eg, 973 of '0' images were correctly predicted as 0, etc)
- Precision: Of all samples predicted as this digit, how many were actually this digit?  (low = many false positives)
- Recall: Of all samples predicted as this digit, how many were actually this digit?  (low = many false positives)
- F1 score: Harmonic mean of precision & recall (overall balance)
- Support: Number of test examples of that digit

How to interpret results: 
- 98% of all digits were classified correctly 

## CNN Results
```
Epoch 8: train 0.9958, test 0.9985                                              
Best: 0.9985
``` 

## CNN vs. MLP 

- CNN exploits spatial locality: nearby pixels processed together; MLP treats each pixel as independent feature (flat 784 vector)
- CNN use shared weights, fewer params; MLP are fully connected, high parameter count 
- CNN learns edges, then shapes, then digit structure; MLP learns global patterns 
- CNN 99.3-99.6% accuracy; MLP 98-99% accuracy 