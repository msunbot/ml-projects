import torch, torch.nn as nn, torch.optim as optim
from utils import get_device, get_loaders
from tqdm import tqdm

class LogReg(nn.Module): # softmax regression
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)  # (B, 1, 28, 28): shape of the MNIST input batch. B = batch size (# of images processed in one forward pass), 1 = number of channels (MNIST is grayscale, RGB = 3), 28x28 = heigh and width of each image 
    def forward(self, x):
        x = x.view(x.size(0), -1) # each image is 1x28x28, we flatten to 784
        return self.fc(x) #linear layer maps 784 to 10 (one logit per class)
    
# one epoch training loop
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train() 
    total = 0; correct = 0; running = 0.0
    for xb, yb in tqdm(loader, ncols=80, leave=False): 
        # Loop through every mini-batch of (images, labels) from the dataset
        # xb is batch of images (shape[B, 1, 28, 28]); 
        # yb is batch of labels (shape[B], eg [3,0,9,1,...])
        # tqdm() warps the loaders to show live progress bar with ncols (width of bar), leave=False (hides progress bar after epoch finishes)
        xb, yb = xb.to(device), yb.to(device) # moves data from CPU to GPU if available 
        # (B, 10): shape of model outputs (logits). Model outputs 10 numbers per image, each number corresponds to one class score (digits 0-9)
        opt.zero_grad() # Reset the slate before computing new gradients, clears any old gradients stored from last batch 
        logits = model(xb) # Forward pass - send images through the model. Logits is model's raw output scores before softmax. Shape = [batch_size, 10] (one score per digit 0-9)
        loss = loss_fn(logits, yb)  # Compute the loss function. Cross Entropy loss between model output(logits) and true labels (integer class IDs)
        # applies softmax inside, convert logits to probabilities, return scalar tensor (eg, 0.32)
        loss.backward() # backpropagation step: computes gradients of loss wrt each parameter, populate param.grad for every laeyr's weights and biases. 
        # Figure out how to nudge every weight to make loss smaller 
        opt.step() # learning step using an optimizer (eg Adam)
        running += loss.item() * xb.size(0) 
        # loss.item() converts scalar tensor to plain Python float 
        # xb.size(0) = batch size
        # Multiplying them accumulates total loss across all samples for the epoch, to keep track of total accumulated loss so far
        preds = logits.argmax(1) # top 1 class index: argmax(logits, dim=1) = predicted class. ie. take index of the largest score along the class dimension (axis 1)
        # for each sample, pick index of largest logit. that index = predicted digit 
        correct += (preds == yb).sum().item()
        # Count how many predictions matched their labels
        total += xb.size(0)
    return running/total, correct/total # mean loss, accuracy 

# evaluation loop
@torch.no_grad() # disables gradient bookkeeping, faster, less memory
def evaluate(model, loader, loss_fn, device):
    model.eval() # switch model to evaluation model, turn off training behaviors like dropout/batch-norm stats
    total = 0; correct = 0; running = 0.0
    for xb, yb in loader: # iterate over all test batches (no tqdm here)
        xb, yb = xb.to(device), yb.to(device) 
        logits = model(xb) # forward pass only 
        loss = loss_fn(logits, yb) # compute cross entry loss 
        running += loss.item() * xb.size(0) # accumulate the sum of losses over samples
        preds = logits.argmax(1) # pick top 1 class index per sample 
        correct += (preds == yb).sum().item() # count correct predictions 
        total += xb.size(0) # track how many samples we've evaluated
    return running/total, correct/total

# Script entry point
if __name__ == "__main__":
    device = get_device() # pick "cuda" if available, else "cpu"
    train_loader, test_loader = get_loaders(batch_size=256) # create data loaders with preprocessing from utils.py
    model = LogReg().to(device) # instatiate softmax regression model and move its parameters to device 
    loss_fn = nn.CrossEntropyLoss() # loss expects raw logits ([B, 10]) and integer labels ([B], dtype torch.long)
    opt = optim.Adam(model.parameters(), lr=1e-3) # Optimizer that will update params using Adam with learning rate of 1e-3 

    best_acc = 0.0 # keep track of best test accuracy seen so far 
    for epoch in range(5): 
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: train acc={tr_acc:.4f} test acc={te_acc:.4f}")
        if te_acc > best_acc: # if test accuracy improve, update best_acc and save checkpoint of model weights
            best_acc = te_acc
            torch.save(model.state_dict(), "mnist_logreg.pt")
    print(f"Best test acc: {best_acc:.4f}")
    # Print metrics and save weights if the test accuracy improved.
    
