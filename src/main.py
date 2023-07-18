

import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt

from src.problems import LogisticRegression
from src.optimizers import *
from src.optimizers.utils import generate_mixing_matrix

from src.experiment_utils import run_exp



if __name__ == '__main__':
    n_agent = 80
    m = 407
    dim = 123    

    kappa = 100
    mu = 5e-8

    n_iters = 10

    p = LogisticRegression(n_agent=n_agent, m=m, dim=dim, noise_ratio=0.05, graph_type='binomial', kappa=kappa, graph_params=0.4, dataset="a9a", gpu=False, regularization= 'l2', LAMBDA=1e-3)
    print(p.n_edges)


    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)

    W, alpha = generate_mixing_matrix(p) # returns the FDLA mixing matrix(Symmetric fastest distributed linear averaging).
    print('alpha = ' + str(alpha))
    

    eta = 2/(p.L + p.sigma)
    eta_2 = 2 / (p.L + p.sigma)
    eta_1 = 1 / p.L
    n_inner_iters = int(m * 0.05)
    batch_size = int(m / 10)
    batch_size = 10
    n_iters = 200
    n_inner_iters = 100

    n_dgd_iters = n_iters * 20


    exps = [
        DGD(p, n_iters=n_iters, eta=eta/10, x_0=x_0, W=W),
        DINSerial(p, n_iters=n_iters, x_0=x_0, W=W),
        DINParallel(p, n_iters=n_iters, x_0=x_0, W=W),
        ]

    res = run_exp(exps, name='logistic_regression', n_cpu_processes=4, save=True, plot= True)

    
    plt.show()
