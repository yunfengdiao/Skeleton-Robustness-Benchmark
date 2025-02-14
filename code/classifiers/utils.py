import os
import logging
import numpy as np
import torch
from collections import OrderedDict
class MyAdam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False, initial_decay=0):

        # Arguments
        # lr: float >= 0. Learning rate.
        # beta_1: float, 0 < beta < 1. Generally close to 1.
        # beta_2: float, 0 < beta < 1. Generally close to 1.
        # epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        # decay: float >= 0. Learning rate decay over each update.
        # amsgrad: boolean. Whether to apply the AMSGrad variant of this
        #    algorithm from the paper "On the Convergence of Adam and
        #    Beyond".

        self.iteration = 0
        self.learningRate = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.amsgrad = amsgrad
        self.initial_decay = initial_decay

    def get_updates(self, grads, params):

        rets = torch.zeros(params.shape)

        lr = self.learningRate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * self.iteration))

        t = self.iteration + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        ms = np.zeros(params.shape)
        vs = np.zeros(params.shape)


        if self.amsgrad:
            vhats = np.zeros(params.shape)
        else:
            vhats = np.zeros(params.shape)

        for i in range(0, rets.shape[0]):
            p = params[i].cpu().detach().numpy()
            g = grads[i].cpu().numpy()
            m = ms[i]
            v = vs[i]
            vhat = vhats[i]


            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)

            if self.amsgrad:
                vhat_t = np.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (np.sqrt(vhat_t) + self.epsilon)
                vhat = vhat_t

            else:
                p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)

            rets[i] = torch.from_numpy(p_t)


        return rets

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def to_labels(y, axis=1):
    """ NxC tensor to NX1 labels """
    return np.argmax(y, axis)


def create_logger(save_path='', file_type='train', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)

def add_into_weights(model, grad_on_weights, gamma):
    names_in_gow = grad_on_weights.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_gow:
                param.add_(gamma * grad_on_weights[name])


def get_grad(model):
    grad_dict = OrderedDict()
    for name, param in model.named_parameters():
        # print(name)
        if param.grad==None:
            continue
        grad_dict[name] = param.grad.data + 0
    return grad_dict


def assign_grad(model, grad_dict):
    names_in_grad_dict = grad_dict.keys()
    for name, param in model.named_parameters():
        if name in names_in_grad_dict:
            if param.grad != None:
                param.grad.data.mul_(0).add_(grad_dict[name])
            else:
                param.grad = grad_dict[name]


def cat_grad(grad_dict):
    dls = []
    for name, d in grad_dict.items():
        dls.append(d)
    return _concat(dls)

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def update_swag_model(model, mean_model, sqmean_model, n):
    for param, param_mean, param_sqmean in zip(model.parameters(), mean_model.parameters(), sqmean_model.parameters()):
        param_mean.data.mul_(n / (n + 1.)).add_(param, alpha=1. / (n + 1.))
        param_sqmean.data.mul_(n / (n + 1.)).add_(param ** 2, alpha=1. / (n + 1.))