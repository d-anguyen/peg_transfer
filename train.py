import torch
import torch.nn as nn
import time 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 20
lr = 1e-3
weight_decay = 1e-4


def train(model, train_loader, val_loader, num_epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            start_time = time.time()
            inputs = inputs.float().to(device)    # [B, T, C, H, W]
            labels = labels.to(device)
                        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if i%25 ==0:
                print(f"Iteration {i}, loss={loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        print(f'Elapsed time for this epoch: {(time.time()-start_time):.2f} seconds')
        print(f'Train accuracy: {evaluate_model(model, train_loader):.2f}%')
        print(f'Validation accuracy: {evaluate_model(model, val_loader):.2f}%')
        print('--------------------------------------------')
        
        
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    #print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy