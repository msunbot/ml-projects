import argparse, os
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# -------- features ----------
def poly_deg2(X): # X: (N,2) numpy 
    x1, x2 = X[:,0], X[:,1]
    return np.stack([x1, x2, x1**2, x1*x2, x2**2], axis=1)

def poly_deg3(X):
    x1, x2 = X[:,0], X[:,1]
    return np.stack([x1, x2, 
                     x1**2, x1*x2, x2**2,
                     x1**3, (x1**2)*x2, x1*(x2**2), x2**3], axis=1)

def standardize_train_test(Xtr, Xva): 
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd==0]=1
    return (Xtr-mu)/sd, (Xva-mu)/sd, mu, sd 

# -------- model ----------
class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1) 
    def forward(self, x):
        return self.lin(x).squeeze(1) # logits, no sigmoid 

# -------- training ----------
def train_torch(deg=2, lr=1e-2, lam=1e-2, epochs=1500, batch=128, seed=0, noise=0.25):
    torch.manual_seed(seed); np.random.seed(seed)

    # data
    X, y = make_moons(n_samples=1000, noise=noise, random_state=0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    fmap = poly_deg3 if deg==3 else poly_deg2
    Xtr, Xva = fmap(Xtr), fmap(Xva)
    Xtr, Xva, mu, sd = standardize_train_test(Xtr, Xva)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    model = LogReg(d=Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=lam)
    loss_fn = nn.BCEWithLogitsLoss() # combines a Sigmoid layer and the Binary Cross Entropy loss into a single, numerically stable class

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)

    print_every = 100

    best_acc, best_state = -1, None 
    for e in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb) 
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        
        if e % print_every == 0 or e == 1: 
            model.eval()
            with torch.no_grad():
                # train metrics
                tr_probs = torch.sigmoid(model(Xtr_t))
                tr_preds = (tr_probs > 0.5).float()
                tr_acc = (tr_preds == ytr_t).float().mean().item()

                # val metrics
                va_probs  = torch.sigmoid(model(Xva_t))
                va_preds  = (va_probs > 0.5).float()
                va_acc    = (va_preds == yva_t).float().mean().item()
            print(f"Epoch {e:4d} | train acc={tr_acc:.4f} val acc={va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc 
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()} 

    if best_state is not None: 
        model.load_state_dict(best_state)
    
    return model, (mu, sd), fmap, best_acc, (X, y)
# -------- plotting ----------
def plot_boundary(model, fmap, mu, sd, X, y, outpath):
    device = next(model.parameters()).device
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300),
        np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 300)
    )
    Xg = np.c_[xx.ravel(), yy.ravel()]
    Xg_pf = fmap(Xg)
    Xg_std = (Xg_pf - mu) /sd
    with torch.no_grad():
        logits = model(torch.tensor(Xg_std, dtype=torch.float32, device=device))
        Z = torch.sigmoid(logits).reshape(xx.shape).cpu().numpy()

    plt.figure(figsize=(7,5))
    plt.contourf(xx, yy, (Z>0.5), alpha=0.20, cmap="coolwarm")
    plt.scatter(X[:,0], X[:,1], c=y, s=12, cmap="coolwarm", edgecolors="k", linewidths=0.3)
    plt.title(f"PyTorch Logistic Regression (deg={3 if fmap==poly_deg3 else 2})")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    print(f"Saved plot -> {outpath}")
    plt.show() 

# -------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=3, choices=[2,3], help="polynomial degree (2 or 3)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=1e-2, help="weight decay (L2)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="figs/logreg_torch_boundary.png")
    args = ap.parse_args()

    model, (mu, sd), fmap, best_acc, (X, y) = train_torch(
        deg=args.deg, lr=args.lr, lam=args.lam, epochs=args.epochs,
        batch=args.batch, seed=args.seed, noise=args.noise
    )
    print(f"Best val acc: {best_acc:.4f}")
    plot_boundary(model, fmap, mu, sd, X, y, args.out)
