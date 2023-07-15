

import numpy as np


import numpy as xp
import multiprocessing as mp

from src import log
from src.problems import Problem
from src.optimizers.utils import NAG, StandardNewton


def logit_1d(X, w):
    # return  1/ (1 + np.expm1(-X.dot(w)))
    # return X.dot(w) - np.log(1 + np.exp(X.dot(w)))
    z = X.dot(w).clip(-100, 100)
    return 1 / (1 + np.exp(-z))


def logit_1d_np(X, w):
    return 1 / (1 + np.exp(-X.dot(w)))


def logit_2d(X, w):
    tmp = xp.einsum('ijk,ki->ij', X, w)
    return 1 / (1 + xp.exp(-tmp))


class LogisticRegression(Problem):
    '''f(w) =  - 1 / N * (\sum y_i log(1/(1 + exp(w^T x_i))) + (1 - y_i) log (1 - 1/(1 + exp(w^T x_i)))) + \frac{\lambda}{2} \| w \|^2 + \alpha \sum w_i^2 / (1 + w_i^2)'''
    def grad_g(self, w):
        if self.alpha == 0:
            return 0
        return 2 * self.alpha * w / ((1 + w**2)**2)

    def g(self, w):
        if self.alpha == 0:
            return 0
        return (1 - 1 / (1 + w ** 2)).sum() * self.alpha

    def __init__(self, kappa=None, noise_ratio=None, LAMBDA=0, alpha=0.001, regularization= None, **kwargs):

        self.noise_ratio = noise_ratio
        self.kappa = kappa
        self.alpha = alpha
        self.LAMBDA = LAMBDA

        super().__init__(**kwargs)

        if alpha == 0:
            if kappa == 1:
                self.LAMBDA = 100
            elif kappa is not None:
                self.LAMBDA = 1 / (self.kappa - 1)
            self.L = 1 + self.LAMBDA
            self.sigma = self.LAMBDA if self.LAMBDA != 0 else None
        else:
            self.L = 1 + self.LAMBDA + 6 * self.alpha
            self.sigma = self.LAMBDA + 2 * self.alpha

       
        
        log.info('Initializing using CPU')
        norm, self.x_min, self.f_min = self._init()

        self.X_train /= norm
        self.X_test /= norm
        self.X /= norm

        log.info('Initialization done')


    def _init(self, result_queue=None):

        if xp.__name__ == 'cupy':
            self.cuda()

        log.info('Computing norm')
        norm = xp.linalg.norm(self.X_train, 2) / (2 * xp.sqrt(self.m_total)) # Upper bound of the hessian
        # self.X_train /= norm
        # self.X /= norm

        if self.kappa is not None:
            log.info('Computing min')
            # x_min, count = NAG(self.grad, xp.random.randn(self.dim), self.L, self.sigma, n_iters=1, eps=1e-10)
            x_min, count = StandardNewton(self, x0= xp.random.randn(self.dim), n_iters=100, eps=1e-6)            
            # count = 0
            # x_min = xp.ones((self.m_total, self.dim))

            log.info(f'Standard Newton ran for {count} iterations')
            f_min = self.f(x_min)
            # f_min = 0.112335018
            log.info(f'f_min = {f_min}')
            # log.info(f'grad_f(x_min) = {xp.linalg.norm(self.grad(x_min))}')

        if xp.__name__ == 'cupy':
            norm = norm.item()
            if self.kappa is not None:
                x_min = x_min.get()
                f_min = f_min.item()
            else:
                x_min = f_min = None

        if result_queue is not None:
            result_queue.put(norm)
            if self.kappa is not None:
                result_queue.put(x_min)
                result_queue.put(f_min)

        if self.kappa is not None:
            return norm, x_min, f_min

        return norm, None, None

    def generate_data(self):
        def _generate_data(m_total, dim, noise_ratio, m_test=None):
            if m_test is None:
                m_test = int(m_total / 10)

            # Generate data
            X = np.random.randn(m_total + m_test, dim)

            # Generate labels
            w_0 = np.random.rand(dim)
            Y = logit_1d_np(X, w_0)
            Y[Y > 0.5] = 1
            Y[Y <= 0.5] = 0

            X_train, X_test = X[:m_total], X[m_total:]
            Y_train, Y_test = Y[:m_total], Y[m_total:]

            noise = np.random.binomial(1, noise_ratio, m_total)
            Y_train = (noise - Y_train) * noise + Y_train * (1 - noise)
            return X_train, Y_train, X_test, Y_test, w_0

        self.X_train, self.Y_train, self.X_test, self.Y_test, self.w_0 = _generate_data(self.n_agent * self.m, self.dim, self.noise_ratio)

    def grad_h(self, w, i=None, j=None):
        '''Gradient of h(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
        1. If w is a vector of shape (dim,)
            1.1 If i is None and j is None
                returns the full gradient.
            1.2 If i is not None and j is None
                returns the gradient at the i-th agent.
            1.3 If i is None and j is not None
                returns the i-th gradient of all training data.
            1.4 If i is not None and j is not None
                returns the gradient of the j-th data sample at the i-th agent.
            Note i, j can be integers, lists or vectors.
        2. If w is a matrix of shape (dim, n_agent)
            2.1 if j is None
                returns the gradient of each parameter at the corresponding agent
            2.2 if j is not None
                returns the gradient of each parameter of the j-th sample at the corresponding agent.
            Note j can be lists of lists or vectors.
        '''

        if w.ndim == 1:
            if type(j) is int:
                j = [j]
            if i is None and j is None:  # Return the full gradient
                return self.X_train.T.dot(logit_1d(self.X_train, w) - self.Y_train) / self.m_total + w * self.LAMBDA
            elif i is not None and j is None:
                return self.X[i].T.dot(logit_1d(self.X[i], w) - self.Y[i]) / self.m + w * self.LAMBDA
            elif i is None and j is not None:  # Return the full gradient
                return self.X_train[j].T.dot(logit_1d(self.X_train[j], w) - self.Y_train[j]) / len(j) + w * self.LAMBDA
            else:  # Return the gradient of sample j at machine i
                return (logit_1d(self.X[i][j], w) - self.Y[i][j]).dot(self.X[i][j]) / len(j) + w * self.LAMBDA

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                tmp = logit_2d(self.X, w) - self.Y
                return xp.einsum('ikj,ik->ji', self.X, tmp) / self.m + w * self.LAMBDA
            elif i is None and j is not None:  # Return the stochastic gradient
                res = []
                for i in range(self.n_agent):
                    if type(j[i]) is int:
                        samples = [j[i]]
                    else:
                        samples = j[i]
                    res.append(self.X[i][samples].T.dot(logit_1d(self.X[i][samples], w[:, i]) - self.Y[i][samples]) / len(samples) + w[:, i] * self.LAMBDA)
                return xp.array(res).T
            else:
                log.fatal('For distributed gradients j must be None')
        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None, split='train'):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            if w.ndim > 1 or i is not None or j is not None:
                log.fatal("Function value on test set only applies to one parameter vector")
            X = self.X_test
            Y = self.Y_test

        if i is None:  # Return the function value
            tmp = X.dot(w)
            return -xp.sum(
                (Y - 1) * tmp - xp.log1p(xp.exp(-tmp))
            ) / X.shape[0] + xp.sum(w**2) * self.LAMBDA / 2

        elif j is None:  # Return the function value in machine i
            tmp = self.X[i].dot(w)
            return -xp.sum((self.Y[i] - 1) * tmp - xp.log1p(xp.exp(-tmp))) / self.m + xp.sum(w**2) * self.LAMBDA / 2
        else:  # Return the gradient of sample j in machine i
            tmp = self.X[i][j].dot(w)
            return -((self.Y[i][j] - 1) * tmp - xp.log1p(xp.exp(-tmp))) + xp.sum(w**2) * self.LAMBDA / 2

    def accuracy(self, w, split='train'):

        if len(w.shape) > 1:
            w = w.mean(axis=1)
        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            X = self.X_test
            Y = self.Y_test
        else:
            log.fatal('Data split %s is not supported' % split)

        Y_hat = X.dot(w)
        Y_hat[Y_hat > 0] = 1
        Y_hat[Y_hat <= 0] = 0
        return xp.mean(Y_hat == Y)

    # def hessian(self, w, i=None, j=None):
    #     '''Hessian of h(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
    #     1. If w is a vector of shape (dim,)
    #         1.1 If i is None and j is None
    #             returns the full Hessian.
    #         1.2 If i is not None and j is None
    #             returns the Hessian at the i-th agent.
    #         1.3 If i is None and j is not None
    #             returns the i-th Hessian of all training data.
    #         1.4 If i is not None and j is not None
    #             returns the Hessian of the j-th data sample at the i-th agent.
    #         Note i, j can be integers, lists or vectors.
    #     2. If w is a matrix of shape (dim, n_agent)
    #         if i and j are None
    #             returns the full Hessian.
    #         2.1 if j is None
    #             returns the Hessian of each parameter at the corresponding agent
    #         2.2 if j is not None
    #             returns the Hessian of each parameter of the j-th sample at the corresponding agent.
    #         Note j can be lists of lists or vectors.
    #     '''
    #     # import pdb; pdb.set_trace()
    #     if w.ndim == 1:
    #         if type(j) is int:
    #             j = [j]
    #         if i is None and j is None:  # Return the full hessian

    #             return np.matmul(self.X_train.T.dot(np.multiply(logit_1d(self.X_train, w), 1 - logit_1d(self.X_train, w)))
    #                                 , self.X_train) / self.m_total + self.LAMBDA * np.eye(self.dim)
    #         elif i is not None and j is None:
    #             return np.matmul(self.X_train[i].T.dot(np.multiply(logit_1d(self.X_train[i], w), 1 - logit_1d(self.X_train[i], w)))
    #                                 , self.X_train[i]) / self.m + self.LAMBDA * np.eye(self.dim)
    #         elif i is None and j is not None:  # Return the full hessian
    #             return np.matmul(self.X_train[j].T.dot(np.multiply(logit_1d(self.X_train[j], w), 1 - logit_1d(self.X_train[j], w)))
    #                                 , self.X_train[i]) / len(j) + self.LAMBDA * np.eye(self.dim)
    #         else:  # Return the hessian of sample j at machine i
    #             return np.matmul(self.X_train[i][j].T.dot(np.multiply(logit_1d(self.X_train[i][j], w), 1 - logit_1d(self.X_train[i][j], w)))
    #                                 , self.X_train[i][j]) / len(j) + self.LAMBDA * np.eye(self.dim)
        

    #     elif w.ndim == 2:
    #         if i is None and j is None:
    #             # return the distributed hesssian
    #             return np.matmul(self.X.T.dot(np.multiply(logit_2d(self.X, w), 1 - logit_2d(self.X, w)))
    #                                 , self.X) / self.m_total + self.LAMBDA * np.eye(self.dim)
    #         elif i is not None and j is None:
    #             return np.matmul(self.X[i].T.dot(np.multiply(logit_2d(self.X[i], w), 1 - logit_2d(self.X[i], w)))
    #                                 , self.X[i]) / self.m + self.LAMBDA * np.eye(self.dim)
    #         elif i is None and j is not None:
    #             return np.matmul(self.X[j].T.dot(np.multiply(logit_2d(self.X[j], w), 1 - logit_2d(self.X[j], w)))
    #                                 , self.X) / len(j) + self.LAMBDA * np.eye(self.dim)
            
    #     else:
    #         log.fatal('The dimension of w is not supported')

    def f(self, w, i=None, j=None, split='train'):
        
        # if split == 'train':
        #     X = self.X_train
        #     Y = self.Y_train
        # elif split == 'test':
        #     if w.ndim > 1 or i is not None or j is not None:
        #         log.fatal("Function value on test set only applies to one parameter vector")
        #     X = self.X_test
        #     Y = self.Y_test

        # if i is None:  # Return the function value
        #     tmp = X.dot(w)
        #     return -xp.sum(
        #         (Y - 1) * tmp - xp.log1p(xp.exp(-tmp))
        #     ) / X.shape[0] + xp.sum(w**2) * self.LAMBDA / 2

        # elif j is None:  # Return the function value in machine i
        #     tmp = self.X[i].dot(w)
        #     return -xp.sum((self.Y[i] - 1) * tmp - xp.log1p(xp.exp(-tmp))) / self.m + xp.sum(w**2) * self.LAMBDA / 2
        # else:  # Return the gradient of sample j in machine i
        #     tmp = self.X[i][j].dot(w)
        #     return -((self.Y[i][j] - 1) * tmp - xp.log1p(xp.exp(-tmp))) + xp.sum(w**2) * self.LAMBDA / 2
        

        if split == 'train':
            X = self.X_train
            Y = self.Y_train

        elif split == 'test':
            if w.ndim > 1 or i is not None:
                log.fatal("Function value on test set only applies to one parameter vector")
            X = self.X_test
            Y = self.Y_test

        if i is None:  # Return the function value
            z = -Y*X.dot(w).clip(-100, 100)
            return np.log(1+np.exp(z)).mean() / self.m_total + self.LAMBDA * np.linalg.norm(w, 2)**2 / 2
            
        else:  # Return the function value in machine i
            # tmp = self.X[i].dot(w)
            X = self.X[i, :, :]
            Y = self.Y[i, :]
            z = -Y*X.dot(w)#.clip(-100, 100)
            return np.log(1+np.exp(z)).mean() / self.m + self.LAMBDA * np.linalg.norm(w, 2)**2 / 2

    def grad_new(self, w, i= None, j= None):

        if i is not None:
            X = self.X[i, :, :]
            Y = self.Y[i, :]
            y_hat = logit_1d(X, w)
            grad = X.T.dot(y_hat - Y) / self.m + self.LAMBDA * w
            return grad
        
        else:
            X = self.X_train
            Y = self.Y_train
            y_hat = logit_1d(X, w)
            grad = X.T.dot(y_hat - Y) / self.m_total+ self.LAMBDA * w
            return grad


    def hessian(self, w, i=None, j=None):

        if i is not None:
            X = self.X[i, :, :]
            y_hat = logit_1d(X, w)
            S = np.diag(y_hat * (1-y_hat))
            h = np.dot(np.dot(X.T, S), X) / self.m + self.LAMBDA * np.eye(w.shape[0])
            return h
        
        else:
            X = self.X_train
            y_hat = logit_1d(X, w) 
            S = np.diag(y_hat * (1-y_hat))
            h = np.dot(np.dot(X.T, S), X) / self.m_total + self.LAMBDA * np.eye(w.shape[0])
            return h




