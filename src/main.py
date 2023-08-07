from data import A9ADataset, ClientDataset
from model import LogisticRegression
from utils import (split_data, 
                   standard_newton, 
                   DIN, 
                   loss_function, 
                   collate_fn)

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

dataset = A9ADataset('data/LibSVM/a9a/a9a')
loaded_data = DataLoader(dataset, batch_size=32 ,shuffle=True, drop_last=True)
# import pdb; pdb.set_trace()
# best_model = standard_newton(1, loaded_data, model, criterion, 1e-3, 1e-3)
# f_min = loss_function(best_model, loaded_data, criterion, 1e-3)
f_min = 0.2
###log.info("The minimum value of the loss function is {}".format(f_min))

gaps = []

# import pdb; pdb.set_trace()
from utils import gradient_cheking
#gradient_cheking(loaded_data, model, criterion, 1e-3)
clients_models = DIN(dataset, neighbour_set, n_clients, rho, k)

avg_wights = torch.stack([item.linear.weight.data for item in clients_models]).mean(axis=0)
avg_models = LogisticRegression(123, 1)
avg_models.linear.weight.data = avg_wights

f_avg = loss_function(avg_models, loaded_data, criterion, 1e-3)
optimality_gap =  f_avg - f_min
gaps.append(optimality_gap.item())
#log.info("optimality gap is {}".format(optimality_gap))

# plot the optimality gap vs iteration
plt.plot(gaps)
plt.xlabel('Communication round')
plt.ylabel('optimality gap')
plt.show()


