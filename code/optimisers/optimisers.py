import torch
from torch.optim import Optimizer
from copy import deepcopy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SGHMC(Optimizer):
    def __init__(self, params, config):
        defaults = dict(lr=0.5, alpha=0, gamma=0.01, L=15, T=1e-5)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGHMC, self).__init__(params, defaults)
        self.config = config

    def __setstate__(self, state):
        super(SGHMC, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # parameter setting
        c = self.config
        L = int(c['L'])  # number of Langevin steps
        gamma = c['gamma']  # 0.01
        lr = c['lr']  # lr=0.5
        T = c['T']  # 1e-5
        # inner sampling steps
        for i in range(L):
            lr = float(lr / np.sqrt(L + 1))

            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    d_param = param.grad.data

                    param_state = self.state[param]
                    if 'mometum' not in param_state:
                        momentum = param_state['momentum'] = torch.zeros(param.data.size()).cuda()
                        noise = deepcopy(momentum)

                    noise.normal_()
                    momentum = (1 - gamma) * momentum - lr * d_param + float(np.sqrt(2 * lr * gamma * T)) * noise
                    param.data.add_(momentum)

        return loss


'''
Stochastic Adaptive HMC
'''

class SGAdaHMC(Optimizer):
    def __init__(self, params, config):
        defaults = dict(lr=0.001, alpha=0, gamma=0.01, L=30, T=1e-5, tao=2, C=1)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(SGAdaHMC, self).__init__(params, defaults)
        self.config = config

    def __setstate__(self, state):
        super(SGHMC, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # parameter setting
        c = self.config
        L = int(c['L'])  # number of Langevin steps
        gamma = c['gamma']  # 0.01
        lr = c['lr']  # lr=0.5
        T = c['T']  # 1e-5

        epsilon = lr
        tao = c['tao']
        C = c['C']

        # inner sampling steps
        for i in range(L):
            # for i in range(randint(1,L)*2):
            lr = float(lr / np.sqrt(L + 1))
            #f = closure()

            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    d_param = param.grad.data

                    param_state = self.state[param]
                    # if 'mometum' not in param_state:
                    if True:
                        momentum = param_state['momentum'] = torch.zeros(param.data.size()).to(device)
                        v = param_state['v'] = 1e-5 * torch.ones(param.data.size()).to(device)
                        noise = deepcopy(momentum)

                    # generate noise
                    noise.normal_()
                    # update scoring matrix v
                    # print('v', v.shape, ' d_param', d_param.shape, 'd_param**2', (torch.mul(d_param, d_param)).shape)
                    v = v - 1 / tao * v + 1 / tao * torch.mul(d_param, d_param)

                    # print(v)
                    # update momentum
                    momentum = momentum - epsilon ** 2 * (1 / torch.sqrt(v)) * d_param - epsilon * (
                                1 / torch.sqrt(v)) * C * momentum + T * (2 * epsilon ** 3 * (1 / torch.sqrt(v)) * C * (
                                1 / torch.sqrt(v)) - epsilon ** 4) * noise

                    param.data.add_(momentum)

        return loss