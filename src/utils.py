
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from data import A9ADataset
from model import LogisticRegression



def enable_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


# def hessian(data_loader, model, criterion):
#     """
#     Compute the Hessian of the loss w.r.t. the model parameters per sample.
#     """
#     model.train()  # Set the model to training mode

#     enable_requires_grad(model, requires_grad=True)  # Enable requires_grad for model parameters

#     hessian = []
#     for data, target in data_loader:
#         output = model(data)
#         loss = criterion(output, target.to(torch.float32))
#         grad1 = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=True, retain_graph=True)
#         param_hessian = []
#         for i in range(len(grad1)):
#             grad2 = torch.autograd.grad(outputs=grad1[i], inputs=model.parameters(), retain_graph=True)
#             param_hessian.append(torch.cat([grad.flatten() for grad in grad2]))
#         hessian.append(torch.stack(param_hessian))
#     hessian = torch.stack(hessian)

#     return hessian

def hessian_by_hand(loader, model, LAMBDA):

    hessian = []
    for batch_idx, (data, target) in enumerate(loader):
        import pdb; pdb.set_trace()
        y_hat = model(data)
        S = torch.diag(y_hat * (1-y_hat))
        h = torch.matmul(torch.matmul(data.T, S), data) + LAMBDA * torch.eye(data.shape[1])
        hessian.append(torch.stack(h))
    return torch.stack(hessian)


from torch.autograd.functional import hessian


a9a = A9ADataset('/kaggle/input/a9a-ahmed/a9a')
a9a_loader = DataLoader(a9a, batch_size=1000, shuffle=True)
model = LogisticRegression(123, 1)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def loss_fun(input_tensor, target_tensor):
    
    model.train()  # Set the model to training mode

    enable_requires_grad(model, requires_grad=True)  # Enable requires_grad for model parameters
    output = model(input_tensor)
    loss = criterion(output, target.to(torch.float32))
    return loss
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import pdb; pdb.set_trace()
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(a9a_loader):
        data, target = data.to(device), target.to(device)
        print(hessian(loss_fun, (data, target)))
        break



