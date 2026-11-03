import torch
import torch.nn as nn
import wandb
from models.customnet import CustomNet
from dataset.tiny_imagenet import get_dataloaders

def train_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in train_loader:
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

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Log delle metriche su WandB
    wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch})
    print(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')

def validate(epoch, model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    # Log su WandB
    wandb.log({"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
    print(f'Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    return val_acc

def main():
    # 1. Inizializza WandB
    wandb.init(project="AML-Lab3", name="customnet-tinyimagenet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data & Model
    train_loader, val_loader = get_dataloaders()
    model = CustomNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_acc = validate(epoch, model, val_loader, criterion, device)
        
        # Salva il miglior checkpoint in checkpoints/
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

if __name__ == '__main__':
    main()