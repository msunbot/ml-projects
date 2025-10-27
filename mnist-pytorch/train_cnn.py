import torch, torch.nn as nn, torch.optim as optim
from utils import get_device, get_loaders
from tqdm import tqdm

# Define CNN architecture
class SmallCNN(nn.Module):
    def __init__(self, pdrop=0.25):
        super().__init__()
        # take an image and detect features
        # first layer: 1 to 32 - learn 32 filters from grayscale
        # second: 32 to 64 - learn higher level patterns (curves, corners)
        # padding=1 keep output size same before pooling 
        # ReLU(): non linearity so netowrk can model complex patterns 
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # downsamples each 2x2 block to its max - halves width and height (28 to 14, then 14 to 7)
            nn.Dropout(pdrop), # randomly drops 25% of activations, a way to regularize to prevent overfitting 

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(256, 10) # for final classification 
        )
    def forward(self, x): 
        x = self.features(x)
        return self.classifier(x)

# training loop wrapper
def run(epochs=8, lr=1e-3, batch=256, wd=1e-4):
    device = get_device()
    train_loader, test_loader = get_loaders(batch)
    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # main training loop 
    best = 0.0 
    for e in range(epochs):
        # --- train ---
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

    # evaluation 
    model.eval(); tot=0; corr=0; loss_sum=0
    with torch.no_grad():
        for xb, yb in test_loader: 
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += loss_fn(logits, yb).item()*xb.size(0)
            corr += (logits.argmax(1)==yb).sum().item()
            tot += xb.size(0)
        te_loss, te_acc = loss_sum/tot, corr/tot

    # save best model
    print(f"Epoch {e+1}: train {tr_acc:.4f}, test {te_acc:.4f}")
    if te_acc > best: 
        best = te_acc
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Best:", best)

if __name__ == "__main__":
    run()