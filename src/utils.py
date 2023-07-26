from data import A9ADataset
from model import LogisticRegression
from utils import split_data, standard_newton, DIN, loss_function
import torch
from torch.utils.data import DataLoader
from  topolgy import Topolgy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import log
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


k = 200
n_clients = 80
rho = 0.1
LAMBDA = 1e-3

criterion = torch.nn.BCELoss()

# set up the topology
topology = Topolgy()
topology.generate_graph(params = 0.2)
neighbour_set =  nx.adjacency_matrix(topology.G).toarray()
neighbour_set = torch.tensor(neighbour_set, dtype=torch.float32)
nodes_degrees = torch.sum(neighbour_set, axis=1)

model = LogisticRegression(123, 1)
clients_models = [model for _ in range(k)]
clients_prev_model = [model for _ in range(k)]

d = torch.randn_like(model.linear.weight.data)
clients_d = [d for _ in range(k)]
prev_d = torch.randn_like(model.linear.weight.data)
clients_prev_d = [prev_d for _ in range(k)]

dual = torch.randn_like(model.linear.weight.data)
clients_dual = [dual for _ in range(k)]

dataset = A9ADataset('data/LibSVM/a9a/a9a')
loaded_data = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

best_model = standard_newton(1, loaded_data, model, criterion, 0.1, 1e-3)
f_min = loss_function(best_model, loaded_data, criterion, 1e-3)
log.info("The value of the loss function is {}".format(f_min))

<<<<<<< HEAD
for i in tqdm(range(5)):
=======
for i in tqdm(range(k)):
>>>>>>> 5195b1877f3cefd78256837975c2a6816387e10d
    index_gen = split_data(dataset, n_clients)
    all_ds = []
    all_prev_ds = []
    for j in range(n_clients):
        ds = torch.sum(clients_d[j], axis= 0)
        prev_ds = torch.sum(clients_prev_d[j], axis= 0)
        all_ds.append(ds)
        all_prev_ds.append(prev_ds)
        
    temp = [0 for _ in range(k)]
    temp_model = [0 for _ in range(k)]
    for j in range(n_clients):
        client_dataset = index_gen.__next__()
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        # current and previous client's neighbours's duals and d's summation
        m, lambda_, direction = DIN(loaded_data,
                                    clients_prev_model[j],
                                    clients_models[j],
                                    clients_dual[j],
                                    clients_d[j],
                                    clients_prev_d[j],
                                    all_prev_ds[j],
                                    all_ds[j],
                                    nodes_degrees[j],
                                    criterion,
                                    LAMBDA,
                                    rho)

        temp[j] = direction
        temp_model[j] = m
        clients_dual[j] = lambda_

    clients_prev_d = clients_d
    clients_d = temp

    clients_prev_model = clients_models
    clients_models = temp_model


    # compute the optimality gap and append to the list
    # gaps = []
    # optimality_gap = torch.norm((clients_models[i].linear.weight.data).mean() - best_model.linear.weight.data)
    # gaps.append(optimality_gap)
    # log.info("optimality gap is {}".format(optimality_gap))

# plot the optimality gap
# plt.plot(gaps)
# plt.xlabel('iteration')
# plt.ylabel('optimality gap')
# plt.show()
