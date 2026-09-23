from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST


def build_dataloaders(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    train_set = PneumoniaMNIST(
        split="train",
        transform=transform,
        download=False
    )

    val_set = PneumoniaMNIST(
        split="val",
        transform=transform,
        download=False
    )

    test_set = PneumoniaMNIST(
        split="test",
        transform=transform,
        download=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader