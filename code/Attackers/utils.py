import numpy as np
import torch
import os
import logging
import torch.nn.functional as F
import torch
import random
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def to_labels(y, axis=1):
    return np.argmax(y, axis)

class MyAdam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False, initial_decay=0):
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


def joint_corruption(input_data):
    device = input_data.device
    out = input_data.clone()

    flip_prob = torch.rand(size=(out.shape[0],), device=device)

    joint_indices = torch.randint(0, 25, (out.shape[0], 15), device=device)
    for i in range(out.shape[0]):
        if flip_prob[i] < 0.5:
            out[i, :, joint_indices[i], :] = 0
        else:
            temp = out[i, :, joint_indices[i], :]
            Corruption = torch.rand(3, 3, device=device) * 2 - 1  
            temp = temp.permute(1, 2, 3, 0).matmul(Corruption)
            temp = temp.permute(3, 0, 1, 2) 
            out[i, :, joint_indices[i], :] = temp

    return out



def pose_augmentation(input_data):
    device = input_data.device
    B, C, T, H, W = input_data.shape
    Shear = torch.eye(3, device=device).repeat(B, 1, 1) + (torch.rand(B, 3, 3, device=device) - 0.5) * 2
    Shear[:, [0, 1, 2], [0, 1, 2]] = 1
    temp_data = input_data.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
    result = torch.bmm(temp_data, Shear.transpose(1, 2))
    result = result.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
    
    return result



def temporal_cropresize(input_data, l_ratio):
    batch_size=input_data.shape[0]
    C, T, V, M = input_data[0].shape
    min_crop_length = 48
    scale = torch.rand(1).item() * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
    temporal_crop_length = min(max(int(60 * scale), min_crop_length), 60)
    start = torch.randint(0, 60 - temporal_crop_length + 1, (1,)).item()
    for i in range(batch_size):
        temporal_context=input_data[i]
        temporal_context = temporal_context[:, start:start + temporal_crop_length, :, :]
        temporal_context = temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_context = temporal_context[None, :, :, None]
        temporal_context = F.interpolate(temporal_context, size=(60, 1), mode='bilinear', align_corners=False)
        temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0)
        temporal_context = temporal_context.contiguous().view(C, V, M, 60).permute(0, 3, 1, 2)
        input_data[i]=temporal_context

    return input_data