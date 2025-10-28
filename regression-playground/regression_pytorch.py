import torch, torch.nn as nn, torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json, pathlib 

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

def load_california():
    data = fetch_california_housing(as_frame=True)
    X = data.data.values.astype(np.float64)
    y = data.target.values.astype(np.float64)
    return X, y 

def standardize_train_test(X_train, X_test): 
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma
    return Xtr, Xte, mu, sigma

class LinearReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d,1) # bias included here 
    def forward(self, x):
        return self.lin(x).squeeze(1) # (N,)

def run(lr=0.005, epochs=200, wd=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    X, y = load_california()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr, Xva, mu, sigma = standardize_train_test(Xtr, Xva)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    # add dataloader (for mini batch)
    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    # train_loader yields (xb, yb) mini batches each epoch 

    model = LinearReg(d=Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    hist = {"tr_loss": [], "va_loss": [], "tr_r2":[], "va_r2":[]}

    N = Xtr_t.size(0)
    for e in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        # full batch for now, can switch to mini batch with a DataLoader 
        for xb, yb in train_loader: 
            opt.zero_grad() # resets gradient, prevents accumulation 
            # yhat = model(Xtr_t)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() *(xb.size(0))
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            yhat_tr = model(Xtr_t).cpu().numpy()
            yhat_va = model(Xva_t).cpu().numpy()
            tr_loss = float(np.mean((yhat_tr - ytr)**2))
            va_loss = float(np.mean((yhat_va - yva)**2))
            tr_r2 = r2_score(ytr, yhat_tr)
            va_r2 = r2_score(yva, yhat_va)

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
        hist["tr_r2"].append(tr_r2); hist["va_r2"].append(va_r2)

        if e % 20 == 0 or e == 1: # print results on first epoch then every 20 
            print (f"Epoch {e:3d} | train MSE={tr_loss:.4f} R2={tr_r2:.3f} | val MSE={va_loss:.4f} R2={va_r2:.3f}")

    pathlib.Path("figs").mkdir(exist_ok=True)
    with open("figs/hist_torch.json","w") as f: json.dump(hist, f)

    # plot 
    plt.figure(); 
    plt.plot(hist["tr_loss"], label="train MSE"); plt.plot(hist["va_loss"], label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("PyTorch MSE"); plt.tight_layout()
    plt.savefig(FIGS / "torch_minibatch_256_mse.png", dpi=160)

    plt.figure(); 
    plt.plot(hist["tr_r2"], label="train R2"); plt.plot(hist["va_r2"], label="val MR2")
    plt.xlabel("epoch"); plt.ylabel("R^2"); plt.legend(); plt.title("PyTorch R^2"); plt.tight_layout()
    plt.savefig(FIGS / "torch_minibatch_256_r2.png", dpi=160)

    print(f"Saved plots to {FIGS}")
    print(f"Final Val R^2 (torch): {hist['va_r2'][-1]:.3f}")

if __name__ == "__main__":
    run()
