import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

def sigmoid(z): return 1/(1+np.exp(-z))
def add_bias(X): return np.c_[np.ones((X.shape[0],1)),X]

def standardize_train_test(Xtr, Xva): 
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1 # if any features has zero sd (eg a constant), set sd = 1 to avoid dividing by zero 
    return (Xtr-mu)/sd, (Xva-mu)/sd, mu, sd

def poly_features_deg2(X): 
    # full quadratic
    x1, x2 = X[:,0], X[:,1]
    return np.stack([x1, x2, x1**2, x1*x2, x2**2], axis=1)

def poly_features_deg3(X):
    # X: (N, 2) with columns x1, x2 
    x1, x2 = X[:,0], X[:,1]
    # linear + full quadratic (including interaction) + degree-3, 9 features
    return np.stack([x1, x2, 
                     x1**2, x1*x2, x2**2,
                     x1**3, (x1**2)*x2, x1*(x2**2), x2**3
                     ], axis=1)

def train_logreg(X, y, lr=0.1, lam=3e-2, epochs=2000, seed=0):
    # lam = lambda
    rng = np.random.default_rng(seed) # initialize random number generator

    # ensure clean arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    Xtr, Xva = poly_features_deg2(Xtr), poly_features_deg2(Xva)

    # sanity checks
    assert Xtr.shape[0] == ytr.shape[0], (Xtr.shape, ytr.shape)
    assert Xva.shape[0] == yva.shape[0], (Xva.shape, yva.shape)

    # standardize using TRAIN stats only 
    Xtr_std, Xva_std, mu, sd = standardize_train_test(Xtr, Xva)

    # add bias after standardization 
    Xtr_b = add_bias(Xtr_std)
    Xva_b = add_bias(Xva_std)

    print("Train design matrix:", Xtr_b.shape)  # expect (800, 6) -> 5 feats + bias
    print("Val design matrix:  ", Xva_b.shape)  # expect (200, 6)
    
    # initialize weights
    w = rng.normal(0, 0.01, Xtr_b.shape[1]) 

    lrs = [0.05, 0.1, 0.2]
    lams = [1e-3, 3e-3, 1e-2]
    best = (-1, None, None) # (acc, lr, lam)

    for LR in lrs: 
        for LAM in lams: 
            w = rng.normal(0, 0.01, Xtr_b.shape[1])

            for e in range(epochs):
                p = sigmoid(Xtr_b @ w)
                # gradient of BCE + L2 (skip bias term)
                g = (Xtr_b.T @ (p-ytr)) / ytr.size 
                g[1:] += 2*lam*w[1:]
                # step
                w -= lr*g
            # evaluate
            p_val = sigmoid(Xva_b @ w)
            acc = accuracy_score(yva, (p_val > 0.5).astype(int))
            print(f"[deg={2}] lr={LR:.3g} lam={LAM:.3g} val acc:{acc:.4f}")
    
            if acc > best[0]:
                best = (acc, LR, LAM)
    print("BEST:",  best)


#        if (e+1) % 200 == 0:
#            va_pred = (sigmoid(Xva_b@w)>0.5).astype(int)
#            acc = accuracy_score(yva, va_pred)
#            print(f"Epoch {e+1}: val acc {acc:.3f}")

        
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    for deg in [2, 3]:
        for C in [10, 30, 100]:
            pipe = make_pipeline(
                PolynomialFeatures(degree=deg, include_bias=False),
                StandardScaler(),
                LogisticRegression(C=C, max_iter=5000)
            )
            pipe.fit(Xtr, ytr)
            acc = (pipe.predict(Xva) == yva).mean()
            print(f"[sklearn] deg={deg} C={C}  val acc={acc:.4f}")

    return w, (mu, sd)

if __name__ == "__main__":

    # Generate data
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
    w, stats = train_logreg(X, y)
    mu, sd = stats

    # boundary plot 
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300), np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5, 300))
    Xg = np.c_[xx.ravel(), yy.ravel()] # Return a contiguous flattened array

    Xg_pf = poly_features_deg2(Xg)
    Xg_std = (Xg_pf - mu) / sd
    Z = sigmoid(add_bias(Xg_std) @ w).reshape(xx.shape)

    plt.contourf(xx, yy, (Z>0.5), alpha=0.2, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap="coolwarm", edgecolor="k")
    plt.title("Logistic Regression Decision Boundary (NumPy, poly feature)")
    plt.savefig(FIGS / "logreg_scratch_poly3.png", dpi=160)
    print(f"Saved plots to {FIGS}")
    plt.show()


