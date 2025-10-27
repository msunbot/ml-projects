import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_device(): # checks if GPU is available 
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(), # converts PIL image to a tensor (0-1 range)
        transforms.Normalize((0.1307,), (0.3081,)) # standardize pixel values so model trains faster
    ])
    # Download MNIST if not already present into ../data/
    # Applies the transform to every image
    # Splits into training and test sets 
    train = datasets.MNIST(root="../data", train=True, download=True, transform=tfm) 
    test = datasets.MNIST(root="../data", train=False, download=True, transform=tfm)

    # Wrap them in DataLoaders (batches data during training)
    # shuffle=True ensures each epoch sees data in a different order (better generalization)
    # num_workers=2 loads data in parallel (speedup)
    # pin_memory= True improves GPU transfer speed (optional optimization)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )