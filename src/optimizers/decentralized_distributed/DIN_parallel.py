

from src.optimizers import Optimizer
from numpy.linalg import pinv, inv
import numpy as np
import networkx as nx

class DINParallel(Optimizer):

    def __init__(self, p, **kwargs):

        super().__init__(p, **kwargs)

        
        self.rho = 0.001
        self.alpha = 0.1

        
        self.adj_matr = nx.adjacency_matrix(self.p.G).toarray()
        self.nodes_degrees = np.sum(self.adj_matr, axis=1)

        self.d = np.zeros_like(self.x)
        self.d_prev = np.zeros_like(self.x)
        self.lambda_ = np.zeros_like(self.x)


        self.ds_prev = self.d.dot(self.adj_matr.T)
        # import pdb; pdb.set_trace()

    # def update(self):
    #     self.comm_rounds += 1
        
    #     node_models = self.x
    #     node_grads = np.zeros_like(node_models)
    #     node_hessians = np.zeros((self.p.dim, self.p.dim, self.p.n_agent))
    #     node_degrees = self.nodes_degrees
    #     node_hessians_alpha = np.zeros_like(node_hessians)

    #     for i in range(self.p.n_agent):
    #         node_model = node_models[:, i]
    #         node_grads[:, i] = self.grad(node_model, i=i)
    #         node_hessians[:, :, i] = self.hessian(node_model, i=i)
    #         node_hessians_alpha[:, :, i] = node_hessians[:, :, i] + ((2 * self.rho * node_degrees[i] + self.alpha) * np.eye(self.p.dim))

    #     ds = self.d.dot(self.adj_matr.T)  # compute the sum of each neighbor's direction of the last iteration

    #     for i in range(self.p.n_agent):
    #         node_degree = node_degrees[i]
    #         node_d = self.d[:, i]
    #         node_ds = ds[:, i]

    #         # Compute the Newton direction for each node separately
    #         node_d = np.linalg.inv(node_hessians_alpha[:, :, i]) @ (node_grads[:, i] - self.lambda_[:, i] + self.rho * (node_degree * node_d + node_ds))
            
    #         # Update the dual variable for each node separately
    #         self.lambda_[:, i] = self.lambda_[:, i] + self.rho * (node_degree * node_d - node_ds)

    #         # Update the local model for each node separately
    #         self.x[:, i] = self.x[:, i] - node_d

    #     #self.x = self.x.dot(self.W)  # Communicate the updated local model

    

    def update(self):
        self.comm_rounds += 1
        
        
        self.ds = self.d.dot(self.adj_matr.T)  # compute the sum of each neighbor's direction of the current iteration
        
        for i in range(self.p.n_agent):
            node_model = self.x[:, i]
            node_grad = self.p.grad_new(node_model, i=i)
            node_hessian = self.p.hessian(node_model, i=i)
            node_degree = self.nodes_degrees[i]
            node_hessian_alpha = node_hessian + ((2 * self.rho * node_degree + self.alpha) * np.eye(node_hessian.shape[0]))

            # Compute the Newton direction for each node separately
            self.d[:, i] = inv(node_hessian_alpha) @ (node_grad - self.lambda_[:, i] + self.rho * (node_degree * self.d_prev[:, i] + self.ds_prev[:, i]))
            
            # Update the dual variable for each node separately
            self.lambda_[:, i] = self.lambda_[:, i] + self.rho * (node_degree * self.d[:, i] - self.ds[:, i])

            # Update the local model for each node separately
            self.x[:, i] = self.x[:, i] - self.d[:, i]

        
        self.ds_prev = self.ds
        self.d_prev = self.d
        # self.x = self.x.dot(self.W)  # Communicate the updated local model
        

