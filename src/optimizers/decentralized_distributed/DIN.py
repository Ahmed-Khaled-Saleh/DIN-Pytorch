

from src.optimizers import Optimizer
from numpy.linalg import pinv, inv
import numpy as np
import networkx as nx

class DINSerial(Optimizer):

    def __init__(self, p, **kwargs):

        super().__init__(p, **kwargs)

        
        self.rho = 0.001
        self.alpha = 0.1
        # import pdb; pdb.set_trace()
        
        self.adj_matr = nx.adjacency_matrix(self.p.G).toarray()
        self.nodes_degrees = np.sum(self.adj_matr, axis=1)

        self.d = np.zeros_like(self.x)
        self.d_prev = np.zeros_like(self.x)
        self.lambda_ = np.zeros_like(self.x)
        self.ds = np.zeros_like(self.x)
        self.ds_prev = np.zeros_like(self.x)

    # def update(self):
    #     self.comm_rounds += 1

    #     self.H = self.hessian(self.x)
    #     self.H_alpha = self.H +  ((2 * 2* self.rho + self.alpha) * np.eye(self.p.dim))


    #     adj_matr = nx.adjacency_matrix(self.p.G).toarray()
    #     degree_mat = np.diag(np.sum(adj_matr, axis=1))
    #     self.d = degree_mat @ self.W

    #     # compute the newton direction
    #     self.d = inv(self.H_alpha) @ (self.grad(self.x) - self.lambda_ +  self.rho *(2 * self.d + self.ds))
    #     #update the dual variable
    #     self.lambda_ = self.lambda_ + self.rho * (2*self.d - self.ds)
    #     #update the local model
    #     self.x = self.x - self.d
    #     #communiate the updated local model
    #     self.x = self.x.dot(self.W)
        
    
    def update(self):
        self.comm_rounds += 1
        
        for i in range(self.p.n_agent):
            
            node_model = self.x[:, i]
            node_grad = self.p.grad_new(node_model, i= i)
            node_hessian = self.p.hessian(node_model, i= i)
            node_degree = self.nodes_degrees[i]
            node_hessian_alpha = node_hessian + ((2 * self.rho * node_degree + self.alpha) * np.eye(node_hessian.shape[0]))
        
            # Compute the Newton direction for each node separately
            self.d[:, i] = inv(node_hessian_alpha) @ (node_grad - self.lambda_[:, i] + self.rho * (node_degree * self.d_prev[:, i] + self.ds_prev[:, i]))
            
            # Update the dual variable for each node separately
            self.lambda_[:, i] = self.lambda_[:, i] + self.rho * (node_degree * self.d[:, i] - self.ds[:, i])

            # Update the local model for each node separately
            self.x[:, i] = self.x[:, i] - self.d[:, i]

            # compute the sum of each neighbours direction of the last iteration
            self.ds_prev = self.ds
            self.ds = self.d.dot(self.adj_matr.T)
            self.d_prev = self.d
            
            

        # Communicate the updated local model
        # self.x = self.x.dot(self.W)
        
