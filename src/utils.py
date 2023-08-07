
import torch
import log
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import ClientDataset
from model import LogisticRegression


def gradient_by_hand(loader, model, criterion, LAMBDA, k = None):
    total_grad = torch.zeros_like(model.linear.weight)
    all_data = torch.tensor([])
    all_targets = torch.tensor([])
    all_y_hat = torch.tensor([])
    for i, (data, target) in enumerate(loader):
        data = (data - data.mean()) / data.std()
        # if k == 20: import pdb;pdb.set_trace()
        target[target < 0] = 0
        target = target.to(torch.float32)
        y_hat = model(data)

        all_data = torch.cat((all_data, data))
        all_targets = torch.cat((all_targets, target))
        all_y_hat = torch.cat((all_y_hat, y_hat))
        # loss = criterion(y_hat, target)
        # loss.backward()
        # # gradient clipping

        # torch.nn.utils.clip_grad_norm_(model.linear.weight, 1.0)
        # total_grad += model.linear.weight.grad
        # total_grad += torch.matmul(data.T, (y_hat - target))

    total_grad = torch.matmul(all_data.T, (all_y_hat - all_targets))
    #import pdb; pdb.set_trace()
    reg_term = 2 * LAMBDA * torch.norm(model.linear.weight)
    total_grad +=  reg_term
    return total_grad

def hessian_by_hand(loader, model, LAMBDA):
    hessian = torch.zeros((model.linear.weight.data.shape[1], model.linear.weight.data.shape[1]))
    all_data = torch.tensor([])
    all_targets = torch.tensor([])
    all_y_hat = torch.tensor([])

    for i, (data, target) in enumerate(loader):
        data = (data - data.mean()) / data.std()
        y_hat = model(data)
        # S = torch.diag(y_hat * (1-y_hat))
        # h = torch.matmul(torch.matmul(data.T, S), data)
        # hessian += h

        all_data = torch.cat((all_data, data))
        all_targets = torch.cat((all_targets, target))
        all_y_hat = torch.cat((all_y_hat, y_hat))

    S = torch.diag(all_y_hat * (1-all_y_hat))
    hessian = torch.matmul(torch.matmul(all_data.T, S), all_data)
    hessian += 2 * LAMBDA * torch.eye(model.linear.weight.data.shape[1])  # Squared L2 regularization term
    return hessian

def standard_newton(n_epochs, loader, model, criterion, LAMBDA, eps):
    #import pdb; pdb.set_trace()
    x = model.linear.weight.data
    #log.info("Running standard newton method")
    for i in tqdm(range(n_epochs)):
        g = gradient_by_hand(loader, model, criterion, LAMBDA, i)
        h = hessian_by_hand(loader, model, LAMBDA)

        delta_x = torch.linalg.solve(h, -g.squeeze())
        x_new = x + delta_x

        # Check for convergence
        if torch.all(torch.abs(x_new - x) < eps):
            break
        if torch.norm(g) ** 2 < eps:
            #log.info("Converged because gradient norm is less than {}".format(eps))
            break

        x = x_new
        model.linear.weight.data = x
        # import pdb; pdb.set_trace()

        # if torch.norm(g) ** 2 < eps:
        #     log.info("Converged because gradient norm is less than {}".format(eps))
        #     return model
        # # upadte weights
        # model.linear.weight.data -= torch.matmul(torch.inverse(h), torch.squeeze(g))
        # # check for convergence
    #import pdb; pdb.set_trace()
    return model

# the value of the loss function
def loss_function(model, loader, criterion, LAMBDA, k =None):
    total_loss = 0
    # import pdb; pdb.set_trace()
    for i, (data, target) in enumerate(loader):
        target[target < 0] = 0
        target = target.to(torch.float32)
        model.zero_grad()
        y_hat = model(data)
        loss = criterion(y_hat, target)
        total_loss += loss.item()
    total_loss += LAMBDA * torch.norm(model.linear.weight.data) ** 2
    # pdb.set_trace()
    return total_loss / len(loader.dataset)

def split_data(data, n_clients, client_index):
    '''
    split the data between 80 clients as follows:
    every client has 407 instances from a total of 32561
    '''
    n_instances = len(data)
    n_instances_per_client = n_instances // n_clients
    if client_index == n_clients - 1:
        return client_index * n_instances_per_client, n_instances
    return client_index * n_instances_per_client, (client_index + 1) * n_instances_per_client

def collate_fn(data):
    '''
    collate function for the dataloader
    '''
    # data is a list of tuples of (feature, label) tensors
    # filtering out empty tensors
    data = list(filter(lambda x: x[0].shape[0] > 0, data))
    features = torch.stack([item[0] for item in data], dim=0)  # Stack all the features into one big tensor
    labels = torch.stack([item[1] for item in data], dim= 0)    # Stack all the labels into one big tensor

    return features, labels

def DIN(dataset,
        neighbour_set,
        n_clients,
        rho,
        k
        ):

    # initializations
    x = {}
    d = {}
    lambda_ = {}
    prev_d = {}
    for client_idx in range(n_clients):
        x[client_idx] = LogisticRegression(123, 1)
        d[client_idx] = torch.randn_like(x[client_idx].linear.weight.data)
        prev_d[client_idx] = torch.randn_like(x[client_idx].linear.weight.data)
        lambda_[client_idx] = torch.randn_like(x[client_idx].linear.weight.data)

    criterion = torch.nn.BCELoss()
    alpha = 0.01
    degree = torch.sum(neighbour_set, axis=1)

    for j in tqdm(range(k)):
        #import pdb; pdb.set_trace()
        for i in range(n_clients):
            # Normalizing the weights
            # x[i].linear.weight.data = x[i].linear.weight.data / torch.norm(x[i].linear.weight.data)

            start, end = split_data(dataset, n_clients, i)
            client_dataset = ClientDataset(dataset[start:end])
            client_loader = DataLoader(client_dataset, batch_size=32)

            g = gradient_by_hand(client_loader, x[i], criterion, 1e-3, j)
            h = hessian_by_hand(client_loader, x[i], 1e-3)

            max_norm = 1.0

            # # Calculate the L2 norm of the gradients
            # grad_norm = torch.norm(g)
            # if grad_norm > max_norm:
            #     g = g * (max_norm / grad_norm)

            I_d = torch.diag_embed(prev_d[i])
            h_alpha = h + (2 * rho * degree[i] + alpha) * I_d
            h_alpha_inv = torch.inverse(h_alpha.squeeze())

            sum_neighbours_prev_d = sum([neighbour_set[i][l] * prev_d[l] for l in range(n_clients)])
            
            d[i] = (h_alpha_inv @ (g - lambda_[i] + (rho * (degree[i] * prev_d[i]) + sum_neighbours_prev_d)).T).T
            
        for i in range(n_clients):
            lambda_[i] = lambda_[i] + rho * (degree[i] * d[i] - sum([neighbour_set[i][l] * d[l] for l in range(n_clients)]))
            x[i].linear.weight.data -=  d[i]
        
        prev_d = d

    return x


def gradient_cheking(loader, model, criterion, LAMBDA, k = None):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
    for epoch in range(100):
        for x, y in loader:
            optimizer.zero_grad()
            y = y.to(torch.float32)
            y[y < 0] = 0
            output = model(x)
            loss = criterion(output, y)
            loss.backward()  # Automatic differentiation

            original_param = model.linear.weight.data.clone()
            original_grad = model.linear.weight.grad.data.clone()

            # Perturb parameter by a small epsilon
            # epsilon = 1e-6
            # model.linear.weight.data += epsilon
            # # output_perturbed = model(x)
            # # loss_perturbed = criterion(output_perturbed, y)

            # Compute finite difference gradient
            numerical_grad = torch.matmul(x.T, (output - y))

            # Restore original parameter
            model.linear.weight.data = original_param
            # Compare gradients
            manual_grad = numerical_grad  # Replace this line with your manual gradient computation
            error = torch.norm(manual_grad - original_grad) / (torch.norm(manual_grad) + torch.norm(original_grad))
            print(f"Gradient Error: {error.item()}")

        # optimizer.step()

    import pdb; pdb.set_trace()

def Gradient_descent(loader, model, criterion, LAMBDA):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
    for i, (data, target) in enumerate(loader):
        model.zero_grad()
        y_hat = model(data)
        target[target < 0] = 0
        loss = criterion(y_hat, target.to(torch.float32))
        loss.backward()
        optimizer.step()
        
    return model
