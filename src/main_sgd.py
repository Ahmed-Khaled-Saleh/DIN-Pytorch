from data import A9ADataset, ClientDataset
from model import LogisticRegression
from utils import (split_data, 
                   standard_newton, 
                   DIN, 
                   loss_function, 
                   collate_fn,
                   Gradient_descent)

import torch
from torch.utils.data import DataLoader
from  topolgy import Topolgy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import log
import random
from copy import deepcopy
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


k = 200
n_clients = 80
rho = 0.01
LAMBDA = 1e-3

criterion = torch.nn.BCELoss()

# set up the topology
topology = Topolgy()
topology.generate_graph(params = 0.2)
neighbour_set =  nx.adjacency_matrix(topology.G).toarray()
neighbour_set = torch.tensor(neighbour_set, dtype=torch.float32)
nodes_degrees = torch.sum(neighbour_set, axis=1)

model = LogisticRegression(123, 1)
clients_models = [LogisticRegression(123, 1) for _ in range(n_clients)]

dataset = A9ADataset('data/LibSVM/a9a/a9a')
loaded_data = DataLoader(dataset, batch_size=32 ,shuffle=True, drop_last=True)

best_model = standard_newton(100, loaded_data, model, criterion, 1e-3, 1e-3)
f_min = loss_function(best_model, loaded_data, criterion, 1e-3)
# f_min = 31
log.info("The minimum value of the loss function is {}".format(f_min))

gaps = []
for i in tqdm(range(k)):

    temp_model = [0] * n_clients

    for j in range(n_clients):
        start, end = split_data(dataset, n_clients, j)

        client_dataset = ClientDataset(dataset[start:end])
        client_loader = DataLoader(client_dataset, batch_size=32)
        temp_model[j] = Gradient_descent(client_loader, clients_models[j], criterion, LAMBDA)

    for j in range(n_clients):
        clients_models[j] = temp_model[j]

    
    avg_wights = torch.stack([item.linear.weight.data for item in clients_models]).mean(axis=0)
    avg_models = LogisticRegression(123, 1)
    avg_models.linear.weight.data = avg_wights

    f_avg = loss_function(avg_models, loaded_data, criterion, 1e-3)
    optimality_gap =  f_avg - f_min
    gaps.append(optimality_gap.item())
    log.info("optimality gap is {}".format(optimality_gap))

# plot the optimality gap vs iteration
plt.plot(gaps)
plt.xlabel('Communication round')
plt.ylabel('optimality gap')
plt.show()
