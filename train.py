import torch
import torch.nn as nn
import time
import tqdm 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('./models', exist_ok = True)



def train(model, train_loader, val_loader, num_epochs, lr=1e-3, weight_decay=1e-4, lr_step=10, lr_gamma=0.2, 
          eval_after=5, save_path='./models/'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss() #think about this later
    
    for epoch in tqdm(range(num_epochs), desc='Training epoch'):
        print(f"#### Learning rate {scheduler.get_last_lr()[0]:.4e} ####")
        model.train()
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            
            inputs = inputs.float().to(device)    # [B, T, C, H, W]
            labels = labels.to(device)
                        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if i%5 ==0:
                print(f"Iteration {i}, Loss={loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f"Time elapsed for the epoch: {(time.time() - start_time):.2f}")
        
        # Evaluation every after every <eval_after> epochs
        if epoch%eval_after==0 or epoch==num_epochs-1: 
            train_loss, train_acc = evaluate_model(model, train_loader)
            val_loss, val_acc = evaluate_model(model, val_loader)
            print(f'Train loss: {train_loss:.4f}% - Validation loss: {val_loss:.2f}%')
            print(f'Train accuracy: {train_acc:.2f}% - Validation accuracy: {val_acc:.2f}%')
            print('---------------------------------------------')
            
        
        

def evaluate_model(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # sum up batch loss

            # For classification: Get predictions & count correct
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100*correct / total

    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, val_acc, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")
    
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_accuracy']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint '{filename}' (epoch {epoch}, val_acc {val_acc:.4f})")
    return epoch, val_acc, loss


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_every=5, resume_path=None):
    best_val_acc = 0.0
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_path is not None:
        start_epoch, best_val_acc, _ = load_checkpoint(model, optimizer, resume_path, device)
        print(f"Resuming from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss, train_acc, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_acc /= total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_acc, total_val = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_loss /= total_val
        val_acc /= total_val
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint every N epochs
        if ((epoch + 1) % save_every == 0) or (epoch == num_epochs - 1):
            checkpoint_path = f"checkpoint_epoch{epoch+1}_acc{val_acc:.4f}.pth"
            save_checkpoint(model, optimizer, epoch+1, val_acc, val_loss, checkpoint_path)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch+1, val_acc, val_loss, "best_model.pth")
            print("Best model updated.")

    print("Training finished.")