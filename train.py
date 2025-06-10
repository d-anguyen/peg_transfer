import torch
import torch.nn as nn
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_loader, val_loader, num_epochs, lr=1e-3, weight_decay=1e-4):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            
            inputs = inputs.float().to(device)    # [B, T, C, H, W]
            labels = labels.to(device)
                        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if i%5 ==0:
                print(f"Loss={loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, time elapsed = {time.time() - start_time}:.2f")
        print(f'Train accuracy: {evaluate_model(model, train_loader):.2f}%')
        print(f'Validation accuracy: {evaluate_model(model, val_loader):.2f}%')
        print('--------------------------------------------')
        
        
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos,labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    #print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy