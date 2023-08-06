import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from topolgy import Topolgy
# Define your DINModel class as a regularized logistic regression
class DINModel(nn.Module):
    def __init__(self, input_dim):
        super(DINModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Define the training and validation functions
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # L2 regularization
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param)**2
        
        # Total loss with L2 regularization
        loss += 0.5 * beta * l2_regularization
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * len(targets)
            num_samples += len(targets)
    
    return total_loss / num_samples

# Define the main training loop for Per-DIN and meta-Per-DIN
def per_din_meta_per_din(alpha, beta, neighbor_sets, train_datasets, val_datasets, S, T, T_0, K_0, rho):
    input_dim = len(train_datasets[0][0][0])  # Assuming the input data shape is (batch_size, input_dim)
    
    # Initialize model and optimizer
    model = DINModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    criterion = nn.BCELoss()

    topology = Topolgy()
    topology.generate_graph(params = 0.2)
    neighbour_set =  nx.adjacency_matrix(topology.G).toarray()
    neighbour_set = torch.tensor(neighbour_set, dtype=torch.float32)
    nodes_degrees = torch.sum(neighbour_set, axis=1)
    
    # Algorithm steps
    theta = {}  # Store the model parameters for each device
    d = {}      # Store the updates for each device
    lambdas = {} # Store the lambda parameters for each device
    delta = {} # store the node degrees
    for device_idx in range(len(train_datasets)):
        theta[device_idx] = model.state_dict()
        d[device_idx] = torch.zeros_like(torch.cat([param.view(-1) for param in model.parameters()]))
        lambdas[device_idx] = torch.zeros_like(torch.cat([param.view(-1) for param in model.parameters()]))
        delta[device_idx] = nodes_degrees[device_idx]

    for t in range(T):
        for device_idx in range(len(train_datasets)):
            model.load_state_dict(theta[device_idx])
            
            # Prepare data and dataloader for the current device
            train_loader = torch.utils.data.DataLoader(train_datasets[device_idx], batch_size=BATCH_SIZE, shuffle=True)
            
            # Perform the DIN updates
            for inputs, targets in train_loader:
                outputs = model(inputs)
                gradient_loss = criterion(outputs, targets)
                
                # Compute gradients and Hessian
                gradients = torch.autograd.grad(gradient_loss, model.parameters(), create_graph=True)
                Hessian = torch.autograd.grad(gradients, model.parameters(), create_graph=True)
                
                Hessian_inv = torch.linalg.inv(Hessian)
                
                # Compute and apply updates
                d[device_idx] = Hessian_inv * (torch.autograd.grad(gradient_loss, model.parameters()) - lambdas[device_idx] + rho * (delta[device_idx] * d[device_idx - 1] + sum([w_i_j * d[j] for j in neighbor_sets[device_idx]])))
                lambdas[device_idx] = lambdas[device_idx - 1] + rho * (delta[device_idx] * d[device_idx] - sum([w_i_j * d[j] for j in neighbor_sets[device_idx]]))
                
                for param, delta_param in zip(model.parameters(), d[device_idx]):
                    param.data -= delta_param
            
            theta[device_idx] = model.state_dict()
    
    return theta

# Example usage:
# Assuming you have prepared the datasets, neighbor sets, hyperparameters, etc.
# per_din_meta_per_din(alpha, beta, neighbor_sets, train_datasets, val_datasets, S, T, T_0, K_0, rho)
