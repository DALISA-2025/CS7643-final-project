import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from final.diffusemix_core import DiffuseMixTensorDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Transforms
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    transform_diffuse = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(*stats), # Input is already tensor
    ])
    
    # Datasets
    cifar_train = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    diffuse_train = DiffuseMixTensorDataset('./data/cifar10_diffusemix/diffusemix_data.pt', transform=transform_diffuse)
    full_dataset = ConcatDataset([cifar_train, diffuse_train])
    
    trainloader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # Model (ResNet18 modified for CIFAR-10)
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_loss = float('inf')
    
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'final/best_model.pth')
            
    print("Done")

if __name__ == "__main__":
    train_model()
