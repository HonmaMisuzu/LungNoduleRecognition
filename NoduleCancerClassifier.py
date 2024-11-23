import torch
import torch.nn as nn

class NoduleCancerClassifier(nn.Module):
    def __init__(self, num_classes, width):
        super(NoduleCancerClassifier, self).__init__()
        self.width = width
        self.conv1 = nn.Conv2d(1, width, kernel_size=3)
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.50)
        self.fc1 = nn.Linear(width*2*30*30, width*4)
        self.dropout2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(width*4, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))             # 64->62
        x = self.pool1(torch.relu(self.conv2(x))) # 62->60->30
        x = self.dropout1(x)
        x = x.view(-1, self.width*2*30*30)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, num_classes, width):
        super(CNNModel, self).__init__()
        self.width = width
        self.conv1 = nn.Conv2d(1, width, kernel_size=3)
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(width*2, width*4, kernel_size=3) # 添加一层卷积层
        self.dropout1 = nn.Dropout(0.50)
        self.fc1 = nn.Linear(width*4*28*28, width*8)
        self.dropout2 = nn.Dropout(0.50)
        self.fc2 = nn.Linear(width*8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))             # 64->62
        x = self.pool1(torch.relu(self.conv2(x))) # 62->60->30
        x = torch.relu(self.conv3(x))             # 30->28
        x = self.dropout1(x)
        x = x.view(-1, self.width*4*28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    