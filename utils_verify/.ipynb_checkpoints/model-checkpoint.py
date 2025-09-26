import torch
import torch.nn as nn
import torch.nn.functional as F

class AC1(nn.Module):
    def __init__(self, input_size):
        super(AC1, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x


class AC2(nn.Module):
    def __init__(self, input_size):
        super(AC2, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x

class AC3(nn.Module):
    def __init__(self, input_size):
        super(AC3, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x


class AC4(nn.Module):
    def __init__(self, input_size):
        super(AC4, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x


class AC5(nn.Module):
    def __init__(self, input_size):
        super(AC5, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x


class AC6(nn.Module):
    def __init__(self, input_size):
        super(AC6, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x


class AC7(nn.Module):
    def __init__(self, input_size):
        super(AC7, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x

class AC8(nn.Module):
    def __init__(self, input_size):
        super(AC8, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x
    
class AC9(nn.Module):
    def __init__(self, input_size):
        super(AC9, self).__init__()
        self.fc1 = nn.Linear(input_size, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.fc5 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x

class AC10(nn.Module):
    def __init__(self, input_size):
        super(AC10, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 5)
        self.fc5 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        self.if_sig = True

    def forward(self, x, all=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x