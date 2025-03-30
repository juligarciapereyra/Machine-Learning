import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random   

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class MLPClassifierModel():
    def __init__(self, params):
        input_dim = params['input_dim']
        hidden_dim1 = params['hidden_dim1']
        hidden_dim2 = params['hidden_dim2']
        output_dim = params['output_dim']
        lr = params['lr']

        self.model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=20, print_epochs=True):
        train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        if X_val is not None and Y_val is not None:
            val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs, labels
                labels -= 1
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = correct / total

            if X_val is not None and Y_val is not None:
                self.model.eval()
                val_running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs, labels
                        labels -= 1
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_running_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_accuracy = correct / total
                if print_epochs:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, '
                          f'Accuracy: {train_accuracy:.4f}, Val Loss: {val_running_loss / len(val_loader):.4f}, '
                          f'Val Accuracy: {val_accuracy:.4f}')

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted += 1
        return predicted.cpu().numpy()


def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False