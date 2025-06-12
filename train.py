import torch
import torch.nn as nn
import torch.optim as optim
import time

def print_and_log(message, log_file='training_log.txt'):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, loss, time_elapsed, filename, log_file):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accuracy': val_acc,
        'loss': loss,
        'time_elapsed': time_elapsed,
    }
    torch.save(checkpoint, filename)
    print_and_log(f"Checkpoint saved: {filename}", log_file)

def load_checkpoint(model, optimizer, scheduler, device, filename, log_file):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_accuracy']
    loss = checkpoint['loss']
    time_elapsed = checkpoint['time_elapsed']
    print_and_log(f"Loaded checkpoint '{filename}' (epoch {epoch}, val_acc {val_acc:.4f})", log_file)
    return epoch, val_acc, time_elapsed

def train(model, train_loader, val_loader, device, num_epochs=20, 
          lr=1e-3, weight_decay=0.0, step_lr=5, gamma_lr=0.5,  
          model_to_resume=None, save_path='./models/',
          save_every=5, print_every=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=gamma_lr)
    
    desc = ('Network architecture: \n' + str(model)+'\n'+
                   'Number of epochs: ' +str(num_epochs) + 
                   ' - Learning rate: ' +str(lr) +
                   ' - Weight decay: ' +str(weight_decay)+
                   ' - StepLR: ' + str(step_lr)+
                   ' - GammaLR: '+ str(gamma_lr) + '\n'+
                   '##########################################################################'
                   )
    log_file = save_path+'training_log.txt'
    print_and_log(desc, log_file)
    
    best_val_acc = 0.0
    start_epoch = 0
    time_elapsed = 0.0
    
    # Load the model to resume training 
    if model_to_resume is not None:
        start_epoch, best_val_acc, time_elapsed = load_checkpoint(model, optimizer, scheduler, model_to_resume, device, log_file)
        print_and_log(f"Resuming from epoch {start_epoch+1}", log_file)
        
    for epoch in range(start_epoch, num_epochs):
        print_and_log(f"\nEpoch {epoch+1}/{num_epochs} - Current Learning Rate: {optimizer.param_groups[0]['lr']}", log_file)
        
        # --- Training ---
        model.train()
        start_time = time.time()
        train_loss, train_acc, total = 0, 0, 0
        for idx, (inputs, labels) in enumerate(train_loader):
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
            
            if (idx+1) % print_every == 0:
                print_and_log(f" --- Iteration {idx+1} - MinibatchLoss {loss.item():.4f}", log_file)
        train_loss /= total
        train_acc /= total
        print_and_log(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", log_file)

        
        # --- Scheduler step ---
        scheduler.step()
        
        # Compute elapsed time
        epoch_time = time.time()-start_time
        print_and_log(f"Epoch time elapsed: {(epoch_time/3600):.2f} hours", log_file)
        time_elapsed+=epoch_time
        
        # --- Evaluate and save checkpoint every N epochs ---
        if ((epoch + 1) % save_every == 0) or (epoch == num_epochs - 1):
            # Validation
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print_and_log(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", log_file)
            
            # Save model
            filename = save_path+ f"checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, val_loss, time_elapsed, filename, log_file)
            
        # --- Save the best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            filename = save_path+"best_model.pth"
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, val_loss, None, filename, log_file)
            print_and_log("Best model updated.", log_file)

    print_and_log(f"Training finished. Total time elapsed: {(time_elapsed/3600):.2f} hours", log_file)
    

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


