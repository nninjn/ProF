import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def slice_model_at_penultimate_layer(original_model):
    # Slice the model by modifying its forward method to stop at the penultimate layer (i.e., remove the last linear layer).
    linear_layers = original_model.layers
    if len(linear_layers) < 2:
        raise ValueError("Model must have at least two linear layers.")
    layers = []
    for layer in linear_layers[: -1]:
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


model_configs = {
    'ruler': [30, 20, 15, 10, 5, 1],
    'AC1': [16, 8, 1],
    'AC2': [100, 1],
    'AC3': [50, 1],
    'AC4': [100, 100, 1],
    'AC5': [64, 64, 1],
    'AC6': [12, 12, 1],
    'AC7': [64, 32, 16, 8, 4, 1],
    'AC8': [5, 5, 1],
    'AC9': [3, 3, 3, 3, 1],
    'AC10': [5, 5, 5, 5, 1],
    'AC11': [10, 10, 10, 10, 1],
    'AC12': [5, 5, 5, 5, 5, 5, 5, 5, 5, 1],
    'compas1': [16, 8, 1],
    'compas2': [64, 32, 16, 8, 4, 1],
    'compas3': [200, 200, 200, 1],
    'compas4': [10, 10, 10, 10, 10, 10, 10, 10, 10, 1],
    'compas5': [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 1],
    'compas6': [1000, 1000, 1000, 1000, 1],
    'BM0': [64, 16, 1],
    'BM1': [64, 16, 1],
    'BM2': [32, 16, 1],
    'BM3': [100, 1],
    'BM4': [150, 100, 50, 1],
    'BM5': [22, 10, 1],
    'BM6': [9, 9, 1],
    'BM7': [64, 64, 1],
    'BM8': [64, 32, 16, 8, 4, 1],
    'GC1': [50, 1],
    'GC2': [100, 1],
    'GC3': [9, 1],
    'GC4': [6, 4, 1],
    'GC5': [64, 32, 16, 8, 4, 1],
}


class Model(nn.Module):
    def __init__(self, input_size, layers):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.if_sig = True
        self.only_feature = False
        self.sigmoid = nn.Sigmoid()
        
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layer_size))
            else:
                self.layers.append(nn.Linear(layers[i-1], layer_size))
    
    def forward(self, x, all=False):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        if not self.only_feature:
            x = self.layers[-1](x)
            if self.if_sig:
                x = self.sigmoid(x)
        return x

    def get_all(self, x):
        output = [x]
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            output.append(x)
        x = self.layers[-1](x)
        output.append(x)
        return output

def create_model(model_name, input_size):
    if model_name in model_configs:
        return Model(input_size, model_configs[model_name])
    else:
        raise ValueError(f"Model {model_name} not found in configurations.")


def train_model(model, train_loader, device, criterion, epochs) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    model.train()
    for epoch in range(epochs):
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        scheduler.step(train_acc)
        # if(epoch % 50 == 0):
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% ")
    return model