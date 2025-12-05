import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_cifar10_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def create_cifar10_dataloaders(batch_size=128, num_workers=4):
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=get_cifar10_transforms(train=True)
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=get_cifar10_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

