import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# ----------------
# 1) Data
# ----------------

def load_california():
    data = fetch_california_housing(as_frame=True)
    X = data.data.values.astype(np.float64)
    y = data.target.values.astype(np.float64)
    feature_names = list(data.data.columns)
    return X, y, feature_names

def standardize_train_test(X_train, X_test):
    # z-score on train; apply same stats to test
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma
    return Xtr, Xte, mu, sigma

def add_bias(X): 
    # prepend a column of ones to absorb the intercept
    return np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

# ----------------
# 2) Model + loss + gradient
# ----------------
def predict(Xb, w): 
    return Xb @ w # (N,d+1) @ (d+1,) -> (N,)

def mse(yhat, y):
    diff = yhat - y
    return float(np.mean(diff * diff))

def grad_mse(Xb, y, w):
    # gradient wrt w for MSE; shape (d+1)
    N = Xb.shape[0]
    return (2.0/N) * (Xb.T @ (Xb @ w - y))

# ----------------
# 3) Training (batch Gradient Descent)
# ----------------
def train_batch_gd(X, y, lr=0.05, epochs=200, val_split=0.2, seed=0, l2=0.0):
    rng = np.random.default_rng(seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=seed)

    # standardize features
    Xtr, Xva, mu, sigma = standardize_train_test(X_train, X_val)

    # add bias column
    Xtr_b = add_bias(Xtr)
    Xva_b = add_bias(Xva)

    # init weights
    w = rng.normal(0, 0.01, size=Xtr_b.shape[1])

    hist = {"tr_loss":[], "va_loss":[], "tr_r2":[], "va_r2":[]}

    for e in range(1, epochs +1): 
        # forward
        yhat_tr = predict(Xtr_b, w)
        tr_loss = mse(yhat_tr, y_train)

        # L2 regularization (ridge): add l2 * || w_without_bias}}^2 to loss (reporting only)
        if l2 > 0:
            tr_loss += l2 * float(np.sum(w[1:]**2))
        
        # gradient
        g = grad_mse(Xtr_b, y_train, w)
        if l2 > 0:
            g[1:] += 2*l2*w[1:] # don't regularize bias 
        
        # update
        w -= lr * g

        # eval on val
        yhat_va = predict(Xva_b, w)
        va_loss = mse(yhat_va, y_val)

        # metrics
        tr_r2 = r2_score(y_train, yhat_tr)
        va_r2 = r2_score(y_val, yhat_va)

        hist["tr_loss"].append(tr_loss)
        hist["va_loss"].append(va_loss)
        hist["tr_r2"].append(tr_r2)
        hist["va_r2"].append(va_r2)

        if e % 20 == 0 or e == 1:
            print(f"Epoch {e:3d} | train MSE={tr_loss:.4f} R2={tr_r2:.3f} | val MSE={va_loss:.4f} R2={va_r2:.3f}")
    
    return w, (mu, sigma), hist

# ----------------
# 4) Plotting helpers
# ----------------
def plot_curves(hist, prefix="scratch"):
    # Accuracy-ish metric for regression is R2, plot both R2 and MSE
    plt.figure()
    plt.plot(hist["tr_loss"], label="train MSE")
    plt.plot(hist["va_loss"], label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("MSE vs epoch")
    plt.tight_layout()
    plt.savefig(FIGS / f"{prefix}_mse.png", dpi=160)

    plt.figure()
    plt.plot(hist["tr_r2"], label="train R2")
    plt.plot(hist["va_r2"], label="val R2")
    plt.xlabel("epoch"); plt.ylabel("R^2"); plt.legend(); plt.title("R^2 vs epoch")
    plt.tight_layout()
    plt.savefig(FIGS / f"{prefix}_r2.png", dpi=160)

    print(f"Saved plots to {FIGS}")

# ----------------
# 5) Main 
# ----------------
if __name__ == "__main__":
    X, y, feat = load_california()
    w, stats, hist = train_batch_gd(X, y, lr=0.1, epochs=200, l2=1e-3)
    plot_curves(hist, prefix="scratch-lr-0.1")

    # quick report on validation performance 
    print(f"Final Val R^2: {hist['va_r2'][-1]:.3f}")