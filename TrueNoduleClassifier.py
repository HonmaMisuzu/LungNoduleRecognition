import torch
import torch.nn as nn

# 定义模型
class NoduleClassifier(nn.Module):
    def __init__(self):
        super(NoduleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(16 * 254 * 254, 32)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x) #510
        x = self.conv2(x)
        x = torch.relu(x) #508
        x = self.pool(x)  #254
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = x.view(-1, 16 * 254 * 254)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
