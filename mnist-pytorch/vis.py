import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

def load_test():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test = datasets.MNIST(root="../data", train=False, download=True, transform=tfm)
    return test 
# get test dataset with same normalization 

@torch.no_grad()
def show_misclassified(model, weights_path, k=36): # display 36 misclassified images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = load_test()
    # build one big tensor of all test images (xs:[1000,1,28,28]) and labels (ys:[10000])
    xs = torch.stack([test[i][0] for i in range(len(test))]).to(device)
    ys = torch.tensor([test[i][1] for i in range(len(test))]).to(device)

    # load weights, move model to device and set evaluation mode 
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    # Predictions: forward pass all test images
    logits = model(xs)
    preds = logits.argmax(1)
    wrong = (preds != ys).nonzero(as_tuple=True)[0] # prediction not equal to label

    if wrong.numel() == 0:
        print("No misclassifications!"); return 
    
    # Take up to k examples 
    idxs = wrong[:k].cpu().tolist()
    n = len(idxs)
    cols = 6
    rows = (n + cols - 1)// cols
    plt.figure(figsize=(cols*2, rows*2))

    # show each misclassified image with predicted vs. true label 
    for i, idx in enumerate(idxs, 1):
        # unnormalize back to display space (invert the mean/std you applied)
        img = xs[idx].cpu().squeeze().mul(0.3081).add(0.1307)
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(f"pred {int(preds[idx])} / true {int(ys[idx])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from train_mlp import MLP # can swap this to LogReg or SmallCNN later
    print("🔍 Loading model and displaying misclassified examples...")
    try: 
        show_misclassified(MLP(), "mnist_mlp.pt", k=36)
    except FileNotFoundError:
        print("Model weights not found. Make sure 'mnist_mlp.pt' exists in this folder.")
