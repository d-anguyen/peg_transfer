import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 20
lr = 1e-3
weight_decay = 1e-4


def train(model, train_loader, num_epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            
            inputs = inputs.float().to(device)    # [B, T, C, H, W]
            labels = labels.to(device)
                        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if i%50 ==0:
                print(f"Loss={loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy