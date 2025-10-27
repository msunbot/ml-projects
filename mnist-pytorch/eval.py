import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_test():
    tfm = transforms.Compose([transforms.ToTensor(), 
                              transforms.Normalize((0.1307,), (0.3081,))])
    # build standardized test set 
    test = datasets.MNIST(root="../data", train=False, download=True, transform=tfm)
    # converts into 2 big tensors
    xs = torch.stack([test[i][0] for i in range(len(test))])
    ys = torch.tensor([test[i][1] for i in range(len(test))])
    return xs, ys

def eval_model(model, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys = load_test()
    xs = xs.to(device); ys = ys.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        logits = model(xs)
        preds = logits.argmax(1)
    # confusion matrix: count of true vs. predicted per class
    # classification report: precision / recall / F1 per digit 
    cm = confusion_matrix(ys.cpu().numpy(), preds.cpu().numpy())
    print("Confusion matrix:\n", cm)
    print(classification_report(ys.cpu().numpy(), preds.cpu().numpy()))
# Example usage:
# from train_logreg import LogReg; eval_model(LogReg(), "mnist_logreg.pt")
# from train_mlp import MLP;       eval_model(MLP(),     "mnist_mlp.pt")

if __name__ == "__main__":
    from train_mlp import MLP
    eval_model(MLP(), "mnist_mlp.pt")