#!/usr/bin/env python
# coding=utf-8
from src.optimizers import Optimizer


class DGD(Optimizer):

    def __init__(self, p, eta=0.1, **kwargs):

        super().__init__(p, **kwargs)
        self.eta = eta

    def update(self):
        self.comm_rounds += 1
        for i in range(self.p.n_agent):
            self.x[:, i] = self.x[:, i] - self.eta * self.p.grad_new(self.x[:, i], i=i)

