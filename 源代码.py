import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
epochs = 10
learning_rates = [0.1, 0.01, 0.001]  
input_size = 784  # 28*28 MNIST图像展平后维度
hidden_size = 128
num_classes = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 写法1：继承nn.Module自定义类，灵活度高，适合复杂多分支模型
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

# 写法2：nn.Sequential序列式定义，代码更简洁，适合线性堆叠的简单模型
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
        for x, y in train_loader:
            # 核心训练循环，完全贴合你要的标准骨架
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_train_loss.append(avg_loss)

        # 测试集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        best_acc = max(best_acc, test_acc)
        print(f"lr={lr} | Epoch {epoch+1}/{epochs} | 平均训练Loss: {avg_loss:.4f} | 测试集准确率: {test_acc:.4f}")
    
    print(f"学习率lr={lr} 最佳测试准确率: {best_acc:.4f}\n")
    return epoch_train_loss, best_acc

loss_records = {}
acc_records = {}
for lr in learning_rates:
    loss, acc = train_and_evaluate(lr)
    loss_records[lr] = loss
    acc_records[lr] = acc

plt.figure(figsize=(10,6))
for lr in learning_rates:
    plt.plot(range(1, epochs+1), loss_records[lr], label=f"lr={lr}")
plt.xlabel("Epoch")
plt.ylabel("Average Train Loss")
plt.title("不同学习率下的MNIST MLP训练Loss变化")
plt.legend()
plt.grid()
plt.show()
