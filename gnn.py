import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, global_mean_pool

# Load the QM9 dataset
dataset = QM9(root="data/QM9")
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # Increased batch size

# Define the GNN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)  # Predict a single property

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)  # Pooling layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Normalizing the target
target_mean = dataset.data.y.mean(dim=0)
target_std = dataset.data.y.std(dim=0)
dataset.data.y = (dataset.data.y - target_mean) / target_std  # Normalize the target values

# Instantiate the model, loss function, and optimizer
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Keep learning rate small
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y[:, 0].unsqueeze(1))  # Predict a single target feature
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
