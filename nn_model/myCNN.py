import torch.nn as nn

Kinds = 2


class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        # featrue.shape = (32 - 3 + 2 * 1) / 2 + 1 = 16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, Kinds)
        self.dropout = nn.Dropout()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pooling(self.relu(self.conv5(x)))
        x = x.view(-1, 1024)
        # print("forward:", x.shape)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        return x
