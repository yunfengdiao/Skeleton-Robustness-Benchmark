from collections import OrderedDict
import torch
import os
import logging
import torch

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