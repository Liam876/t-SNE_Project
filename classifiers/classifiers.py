import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from torch import nn

# Set seed
seed = 35
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Constants
BATCH_SIZE = 32
VALID_SPLIT = 0.20
EPOCHS = 50
LR = 1e-2

# Paths
features_file = 'csv_files/all_init.csv'  # Path to circle dataset csv file
labels_file = 'csv_files/all_labels.csv'    # Path to circle labels csv file

#############################################################################

class CustomDataset(Dataset):
    """
    Custom dataset class for points and labels.
    """
    def __init__(self, points_file, labels_file, model = "reg",mode = "train"):
        """
       Initializes the CustomDataset with the given feature and label files.

       Args:
           points_file (str): Path to the CSV file containing feature data (points).
           labels_file (str): Path to the CSV file containing label data.
           model (str): The type of model for which the data is being prepared.
                        Use "special" for LSTM and transformers to reshape the points for sequence input.
                        Default is "reg" for regular models.
        """


        flatten_points = pd.read_csv(points_file).values[:,1:]
        self.labels = pd.read_csv(labels_file).values[:,1].flatten()  # Flatten in case labels are a single column with shape (N, 1)
        n_points = int(flatten_points.shape[1]//2)
        n_samples = flatten_points.shape[0]
        train_len = int(n_samples * (1- VALID_SPLIT))
        if mode == "train":
            self.features = torch.tensor(flatten_points[:train_len,:], dtype = torch.float32)
            self.labels = torch.tensor(self.labels[:train_len], dtype=torch.long)
            #print(self.features.shape)
            if model == "special":  # Reshape for LSTM and transformer models
                self.features = self.features.view(-1, n_points, 2)
        else:
            self.features = torch.tensor(flatten_points[train_len:, :], dtype=torch.float32)
            self.labels = torch.tensor(self.labels[train_len:], dtype=torch.long)
            # print(self.features.shape)
            if model == "special":  # Reshape for LSTM and transformer models
                self.features = self.features.view(-1, n_points, 2)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BlobsDataset(Dataset):
    """
    Custom dataset class for points and labels.
    """
    def __init__(self, points_file, labels_file, model = "reg",mode = "train"):
        """
       Initializes the CustomDataset with the given feature and label files.

       Args:
           points_file (str): Path to the CSV file containing feature data (points).
           labels_file (str): Path to the CSV file containing label data.
           model (str): The type of model for which the data is being prepared.
                        Use "special" for LSTM and transformers to reshape the points for sequence input.
                        Default is "reg" for regular models.
        """


        flatten_points = pd.read_csv(points_file).values[1:,:]
        self.labels = pd.read_csv(labels_file).values[1:,0]
        n_points = int(flatten_points.shape[1]//2)
        n_samples = flatten_points.shape[0]
        train_len = int(n_samples * (1- VALID_SPLIT))
        if mode == "train":
            self.features = torch.tensor(flatten_points[:train_len,:], dtype = torch.float32)
            self.labels = torch.tensor(self.labels[:train_len], dtype=torch.long)
            #print(self.features.shape)
            if model == "special":  # Reshape for LSTM and transformer models
                self.features = self.features.view(-1, n_points, 2)
        else:
            self.features = torch.tensor(flatten_points[train_len:, :], dtype=torch.float32)
            self.labels = torch.tensor(self.labels[train_len:], dtype=torch.long)
            # print(self.features.shape)
            if model == "special":  # Reshape for LSTM and transformer models
                self.features = self.features.view(-1, n_points, 2)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]




# Models

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for classification tasks.

    This MLP model consists of a simple neural network with one hidden layer and ReLU activation.

    Attributes:
        fc1 (nn.Linear): First linear layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Output linear layer.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the MLP model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of units in the hidden layer.
            num_classes (int): Number of output classes.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass for the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class LSTMModel(nn.Module):
    """
    LSTM model for sequence classification tasks.

    This model uses LSTM layers to capture temporal dependencies in sequence data, followed by a linear layer for classification.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
        fc (nn.Linear): Output linear layer.
    """
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=1):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Number of input features per sequence element.
            hidden_size (int): Number of units in the LSTM layer.
            num_classes (int): Number of output classes.
            num_layers (int, optional): Number of LSTM layers. Default is 1.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Using the last time step's output
        return out




class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead=1, num_encoder_layers=3, dim_feedforward=128, num_classes=2):
        """
        Transformer model for classification.

        Args:
            input_dim (int): The number of input features.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_classes (int): The number of output classes.
        """
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass for the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = x.permute(1, 0, 2)  # Transformer expects input of shape (seq_len, batch_size, feature)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)
        return x


def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.

    Returns:
        Tuple[float, float]: Test loss and accuracy.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            #print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, save_every=10, model_name="model"):
    """
    Train the neural network model and evaluate it on the test set.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs to train.
        save_every (int): Save the model every 'save_every' epochs.
        model_name (str): Base name for saving the model and metrics.

    Returns:
        None
    """
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_loss_list.append(epoch_loss)
        train_accuracy_list.append(epoch_accuracy)

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f'{model_name}_epoch_{epoch + 1}_Blobs.pth')
            print(f'Model saved at epoch {epoch + 1}')

    # Save metrics
    np.save(f'{model_name}_train_loss_final_Blobs.npy', np.array(train_loss_list))
    np.save(f'{model_name}_train_accuracy_final_Blobs.npy', np.array(train_accuracy_list))
    np.save(f'{model_name}_test_loss_final_Blobs.npy', np.array(test_loss_list))
    np.save(f'{model_name}_test_accuracy_final_Blobs.npy', np.array(test_accuracy_list))

    # Plot metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label='Train Accuracy')
    plt.plot(test_accuracy_list, label='Test Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} Training and Test Accuracy')
    plt.legend()

    plt.show()



# Create dataset and dataloader
train_dataset = BlobsDataset(features_file, labels_file, mode = "train")
seq_train_dataset = BlobsDataset(features_file, labels_file, model= "special", mode = "train")
test_dataset = BlobsDataset(features_file, labels_file, mode = "test")
seq_test_dataset = BlobsDataset(features_file, labels_file, model= "special", mode = "test")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
seq_train_loader = DataLoader(seq_train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
seq_test_loader = DataLoader(seq_test_dataset,batch_size=BATCH_SIZE,shuffle=True)

# print(train_dataset.labels.shape)
# print(train_dataset.features.shape)
# print(test_dataset.labels.shape)
# print(test_dataset.features.shape)
#print(seq_train_dataset.features.shape)

# Define hyperparameters
MLP_input_size = train_dataset.features.shape[1]
special_input_size = 2
hidden_size = 30
num_classes = len(torch.unique(train_dataset.labels))


# Models to train
models = {
    "MLP": MLP(MLP_input_size, hidden_size, num_classes),
    "LSTM": LSTMModel(special_input_size, hidden_size, num_classes),
    "Transformer": TransformerModel(special_input_size)
}

# Train each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if model_name == "MLP":
        train_model(model, train_loader,test_loader, criterion, optimizer, EPOCHS, save_every=10, model_name=model_name)
    else:
        #print(model_name)
        train_model(model,seq_train_loader,seq_test_loader,criterion,optimizer,EPOCHS,save_every=10,model_name = model_name)


