#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from models.resnet32 import build_resnet32_cifar10
from utils.cifar10_dataset import create_cifar10_dataloaders

CONFIGS = [
    {'name': 'sgd_lr0.1_wd1e-4_20ep_mul', 'batch_size': 128, 'num_epochs': 20,
     'learning_rate': 0.1, 'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 1e-4,
     'scheduler': 'multistep', 'lr_milestones': [10, 15], 'lr_gamma': 0.1},
    {'name': 'adam_lr0.001_wd1e-4_10ep_cos', 'batch_size': 128, 'num_epochs': 10,
     'learning_rate': 0.001, 'optimizer': 'adam', 'momentum': 0, 'weight_decay': 1e-4,
     'scheduler': 'cosine'},
    {'name': 'sgd_lr0.1_wd5e-4_10ep_mul', 'batch_size': 128, 'num_epochs': 10,
     'learning_rate': 0.1, 'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 5e-4,
     'scheduler': 'multistep', 'lr_milestones': [5, 8], 'lr_gamma': 0.1},
    {'name': 'sgd_lr0.01_wd5e-4_10ep_mul', 'batch_size': 128, 'num_epochs': 10,
     'learning_rate': 0.01, 'optimizer': 'sgd', 'momentum': 0.9, 'weight_decay': 5e-4,
     'scheduler': 'multistep', 'lr_milestones': [5, 8], 'lr_gamma': 0.1},
]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]', colour='green')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix(loss=f'{running_loss/total:.3f}', acc=f'{100.*correct/total:.1f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device, epoch, total_epochs):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(test_loader, desc=f'Epoch {epoch}/{total_epochs} [Val]', colour='blue')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=f'{test_loss/total:.3f}', acc=f'{100.*correct/total:.1f}%')
    
    return test_loss / len(test_loader), 100. * correct / total


def save_figures(exp_dir, metrics):
    epochs = list(range(1, len(metrics['train_acc']) + 1))
    
    for data, ylabel, fname, colors, loc in [
        ([metrics['train_acc'], metrics['test_acc']], 'Accuracy (%)', 'accuracy.png', ['b-', 'r-'], 'lower right'),
        ([metrics['train_loss'], metrics['test_loss']], 'Loss', 'loss.png', ['b-', 'r-'], 'upper right'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, data[0], colors[0], label='Training', linewidth=2)
        ax.plot(epochs, data[1], colors[1], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc=loc, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        ax.set_xlim(epochs[0], epochs[-1])
        plt.tight_layout()
        fig.savefig(os.path.join(exp_dir, fname), dpi=150)
        plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, 'learning_rate.png'), dpi=150)
    plt.close(fig)


def train_config(config, device):
    name = config['name']
    exp_dir = os.path.join('experiments', name)
    
    if os.path.exists(os.path.join(exp_dir, 'model_best.pth')):
        return None
    
    os.makedirs(exp_dir, exist_ok=True)
    train_loader, test_loader = create_cifar10_dataloaders(batch_size=config['batch_size'], num_workers=4)
    
    model = build_resnet32_cifar10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                              momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    
    if config['scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=config['lr_gamma'])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    metrics = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'lr': []}
    best_acc = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config['num_epochs'])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, epoch, config['num_epochs'])
        scheduler.step()
        
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'best_acc': best_acc},
                       os.path.join(exp_dir, 'model_best.pth'))
    
    torch.save({'model_state_dict': model.state_dict(), 'epoch': config['num_epochs'], 'final_acc': test_acc},
               os.path.join(exp_dir, 'model_final.pth'))
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'best_val_accuracy': best_acc, 'final_val_accuracy': test_acc,
            'epochs_trained': config['num_epochs'],
            'train_acc_history': metrics['train_acc'], 'test_acc_history': metrics['test_acc'],
            'train_loss_history': metrics['train_loss'], 'test_loss_history': metrics['test_loss'],
            'learning_rate_history': metrics['lr'],
        }, f, indent=2)
    
    save_figures(exp_dir, metrics)
    return best_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('experiments', exist_ok=True)
    results = []
    
    for config in CONFIGS:
        best_acc = train_config(config, device)
        if best_acc:
            results.append({'name': config['name'], 'best_acc': best_acc})
    
    if results:
        results.sort(key=lambda x: x['best_acc'], reverse=True)
        with open('experiments/summary.json', 'w') as f:
            json.dump({
                'total_experiments': len(CONFIGS), 'best_model': results[0]['name'],
                'best_accuracy': results[0]['best_acc'], 'all_results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)


if __name__ == '__main__':
    main()

