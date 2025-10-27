import torch, torch.nn as nn, torch.optim as optim
from utils import get_device, get_loaders
from tqdm import tqdm 

# multilayer perceptron 
class MLP(nn.Module): 
    def __init__(self, hidden=256, pdrop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), # Turns images into feature vectors (B, 784)
            nn.Linear(28*28, hidden), # add a hidden layer vs. logreg 
            nn.ReLU(), # non linearity 
            nn.Dropout(pdrop), # regularization 
            nn.Linear(hidden,10) # Output is logits with 10 classes
        )
    def forward(self, x): 
        return self.net(x)

# Training & evaluation loop 
def run(epochs=8, lr=1e-3, batch=256):
    device = get_device()
    train_loader, test_loader = get_loaders(batch)
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr) # AdamW is Adam + decoupled weight decay, slight regularization benefit
    best = 0.0
    for e in range(epochs): # slightly longer default training (8 vs. 5 in logreg)
        # train
        model.train(); tot=0; corr=0; loss_sum=0
        for xb, yb in tqdm(train_loader, ncols=80, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item()*xb.size(0)
            corr += (logits.argmax(1)==yb).sum().item()
            tot += xb.size(0)
        tr_loss, tr_acc = loss_sum/tot, corr/tot
        # eval
        model.eval(); tot=0; corr=0; loss_sum=0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += nn.functional.cross_entropy(logits, yb).item()*xb.size(0)
                corr += (logits.argmax(1)==yb).sum().item()
                tot += xb.size(0)
            te_loss, te_acc = loss_sum/tot, corr/tot
            print(f"Epoch {e+1}L train {tr_acc:.4f}, test {te_acc:.4f}")
            if te_acc > best: 
                best = te_acc
                torch.save(model.state_dict(), "mnist_mlp.pt")
            print("Best:", best)
        
if __name__ == "__main__":
    run()