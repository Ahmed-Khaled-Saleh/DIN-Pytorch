
import torch
import log
from tqdm import tqdm

def gradient_by_hand(loader, model, criterion, LAMBDA):

    total_grad = torch.zeros_like(model.linear.weight)
    for i, (data, target) in enumerate(loader):
        target[target < 0] = 0
        target = target.to(torch.float32)
        model.zero_grad()
        y_hat = model(data)
        # loss = criterion(y_hat, target)
        # loss.backward()
        # # gradient clipping
        # # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # total_grad += model.linear.weight.grad
        total_grad += torch.matmul(data.T, (y_hat - target))

    reg_term = 2 * LAMBDA * torch.norm(model.linear.weight)
    total_grad +=  reg_term
    return total_grad

def hessian_by_hand(loader, model, LAMBDA):
    hessian = torch.zeros((model.linear.weight.data.shape[1], model.linear.weight.data.shape[1]))
    for i, (data, target) in enumerate(loader):
        model.zero_grad()
        y_hat = model(data)
        S = torch.diag(y_hat * (1-y_hat))
        h = torch.matmul(torch.matmul(data.T, S), data)
        hessian += h
    hessian += 2 * LAMBDA * torch.eye(model.linear.weight.data.shape[1])  # Squared L2 regularization term
    return hessian

def standard_newton(n_epochs, loader, model, criterion, LAMBDA, eps):

    x = model.linear.weight.data
    log.info("Running standard newton method")
    for i in tqdm(range(n_epochs)):
        g = gradient_by_hand(loader, model, criterion, LAMBDA)
        h = hessian_by_hand(loader, model, LAMBDA)

        delta_x = torch.linalg.solve(h, -g.squeeze())
        x_new = x + delta_x

        # Check for convergence
        if torch.all(torch.abs(x_new - x) < eps):
            break
        if torch.norm(g) ** 2 < eps:
            log.info("Converged because gradient norm is less than {}".format(eps))
            break

        x = x_new
        model.linear.weight.data = x
        

        # if torch.norm(g) ** 2 < eps:
        #     log.info("Converged because gradient norm is less than {}".format(eps))
        #     return model
        # # upadte weights
        # model.linear.weight.data -= torch.matmul(torch.inverse(h), torch.squeeze(g))
        # # check for convergence
    import pdb; pdb.set_trace()
    return model

# the value of the loss function
def loss_function(model, loader, criterion, LAMBDA, k =None):
    total_loss = 0
    # import pdb; pdb.set_trace()
    for i, (data, target) in enumerate(loader):
        model.zero_grad()
        y_hat = model(data)
        target[target < 0] = 0
        loss = criterion(y_hat, target.to(torch.float32))
        total_loss += loss
    total_loss += LAMBDA * torch.norm(model.linear.weight.data ** 2)
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

from copy import deepcopy
def DIN(client_loader,
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
        alpha= 0.001):

    g = gradient_by_hand(client_loader, cur_client_model, criterion, LAMBDA)
    h = hessian_by_hand(client_loader, cur_client_model, LAMBDA)
    # import pdb; pdb.set_trace()
    I_d = torch.diag_embed(client_d)

    h_alpha = h + (2 * rho * degree + alpha) * I_d
    h_alpha_inv = torch.inverse(h_alpha.squeeze())
    #import pdb; pdb.set_trace()
    temp_model = deepcopy(cur_client_model)

    d = (h_alpha_inv @ (g - dual + (rho * (degree * client_prev_d) + prev_ds)).T).T
    dual_var = dual + rho * (degree * d - cur_ds)
    temp_model.linear.weight.data = cur_client_model.linear.weight.data - d

    return temp_model, dual_var, d


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
