
import torch
import log
from tqdm import tqdm

def gradient_by_hand(loader, model, criterion, LAMBDA):
    total_grad = torch.zeros_like(model.linear.weight)
    for i, (data, target) in enumerate(loader):
        y_hat = model(data)
        target[target == -1] = 0
        loss = criterion(y_hat, target.to(torch.float32))
        loss.backward()
        total_grad += model.linear.weight.grad
        model.zero_grad()
    total_grad +=  2 * LAMBDA * model.linear.weight
    return total_grad

def hessian_by_hand(loader, model, LAMBDA):
    hessian = torch.zeros((model.linear.weight.data.shape[1], model.linear.weight.data.shape[1]))
    for i, (data, target) in enumerate(loader):
        y_hat = model(data)
        S = torch.diag(y_hat * (1-y_hat))
        h = torch.matmul(torch.matmul(data.T, S), data)
        hessian += h
    hessian += 2 * LAMBDA * torch.eye(model.linear.weight.data.shape[1])  # Squared L2 regularization term
    return hessian

def standard_newton(n_epochs, loader, model, criterion, LAMBDA, eps):

    log.info("Running standard newton method")
    for i in tqdm(range(n_epochs)):
        g = gradient_by_hand(loader, model, criterion, LAMBDA)
        h = hessian_by_hand(loader, model, LAMBDA)

        # upadte weights
        model.linear.weight.data -= torch.matmul(torch.inverse(h), torch.squeeze(g))
        # check for convergence
        if torch.norm(g) < eps:
            log.info("Converged because gradient norm is less than {}".format(eps))
            return model
    return model

# the value of the loss function
def loss_function(model, loader, criterion, LAMBDA):
    total_loss = 0
    for i, (data, target) in enumerate(loader):
        y_hat = model(data)
        target[target == -1] = 0
        loss = criterion(y_hat, target.to(torch.float32))
        total_loss += loss
    total_loss += LAMBDA * torch.norm(model.linear.weight.data) ** 2
    return total_loss

def split_data(data, n_clients, client_index):
    '''
    split the data between 80 clients as follows:
    every client has 407 instances from a total of 32561
    '''
    n_instances = len(data)
    n_instances_per_client = n_instances // n_clients
    if client_index == n_clients - 1:
        return data[client_index * n_instances_per_client:]
    return client_index * n_instances_per_client, (client_index + 1) * n_instances_per_client

def collate_fn(data):
    '''
    collate function for the dataloader
    '''
    features, labels = zip(*data)
    batch_features = torch.stack(features)
    batch_labels = torch.stack(labels)
    return batch_features, batch_labels

def DIN(client_loader,
        prev_client_model,
        cur_client_model,
        dual,
        client_d,
        client_prev_d,
        prev_ds,
        cur_ds,
        degree,
        criterion, 
        LAMBDA, 
        rho,
        ):

    g = gradient_by_hand(client_loader, cur_client_model, criterion, LAMBDA)
    h = hessian_by_hand(client_loader, cur_client_model, LAMBDA)

    h_alpha = h + (2 * rho* degree) * torch.eye(cur_client_model.linear.weight.data.shape[0])
    h_alpha_inv = torch.inverse(h_alpha)

    client_d = (h_alpha_inv @ (g - dual + (rho * (degree * client_prev_d) + prev_ds)).T).T
    dual = dual + rho * (degree * client_d - cur_ds)
    cur_client_model.linear.weight.data = prev_client_model.linear.weight.data - client_d

    return cur_client_model, dual, client_d

