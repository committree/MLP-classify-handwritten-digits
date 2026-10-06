# MLP-classify-handwritten-digits
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
epochs = 10
learning_rates = [0.1, 0.01, 0.001]
input_size = 784
hidden_size = 128
num_classes = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#法一：nn.Module
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits

#法二：nn.Sequential
net_seq = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes)
)

def train_and_evaluate(lr):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_train_loss = []
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        epoch_train_loss.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x,y in test_loader:
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        best_acc = max(best_acc, test_acc)
        print(f"lr={lr} | Epoch {epoch + 1}/{epochs} | 平均训练Loss: {avg_loss:.4f} | 测试集准确率: {test_acc:.4f}")

    print(f"学习率lr={lr} 最佳测试准确率: {best_acc:.4f}\n")
    return epoch_train_loss, best_acc
loss_record = {}
acc_record = {}
for lr in learning_rates:
    loss, acc = train_and_evaluate(lr)
    loss_record[lr] = loss
    acc_record[lr] = acc

plt.figure(figsize=(10,6))
for lr in learning_rates:
    plt.plot(range(1,epochs+1), loss_record[lr], label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Average Train Loss")
plt.title("不同学习率下的MNIST MLP训练Loss变化")
plt.legend()
plt.grid()
plt.show()
