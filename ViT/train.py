import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ViT import ViT
from data import get_CIFAR10_data


class Trainer:
    def __init__(self, model, optimizer, device='cpu', scheduler=None):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs):

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()


            print(f'Epoch [{epoch + 1}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        return history


def trian_with_config():

    config = {
        'root': 'data/cifar10/cifar-10-batches-py',
        'num_training': 49000,
        'num_validation': 1000,
        'num_test': 1000,

        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 256,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 512,
        'pool': 'cls',
        'channels': 3,
        'dim_head': 64,
        'dropout': 0.2,
        'emb_dropout': 0.1,


        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0,
        'optimizer': 'Adam',
        'momentum': 0.9, # for SGD only
        'device': 'cuda',

        'exp_name':"exp1"
    }


    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(
        config['root'],
        num_training=config['num_training'],
        num_validation=config['num_validation'],
        num_test=config['num_test']
    )

    print(f'Number of training samples: {len(X_train)}')
    print(f'Number of validation samples: {len(X_val)}')
    print(f'Number of test samples: {len(X_test)}')


    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = ViT(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        pool=config['pool'],
        channels=config['channels'],
        dim_head=config['dim_head'],
        dropout=config['dropout'],
        emb_dropout=config['emb_dropout']
    )


    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer']}")


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    trainer = Trainer(model, optimizer, device=config['device'], scheduler=scheduler)

    print("\n" + "="*80)
    print("CONFIGURATION:")
    print(config)
    print("="*80)

    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, config['num_epochs'])

    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)


    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_results_{config['exp_name']}.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to 'training_results_{config['exp_name']}.png'")


    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
        'test_acc': test_acc
    }, f'vit_cifar10_model_{config['exp_name']}.pth')
    print(f"Model saved to 'vit_cifar10_model_{config['exp_name']}.pth'")


if __name__ == '__main__':
    trian_with_config()