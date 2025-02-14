import numpy as np
import torch

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def to_labels(y, axis=1):
    """ NxC tensor to NX1 labels """
    return np.argmax(y, axis)


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

            # m_t = np.concatenate(((self.beta_1 * m) , (1. - self.beta_1) * g),axis=0)
            # v_t = np.concatenate(((self.beta_2 * v) , (1. - self.beta_2) * np.square(g)),axis=0)


            if self.amsgrad:
                vhat_t = np.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (np.sqrt(vhat_t) + self.epsilon)
                vhat = vhat_t

            else:
                p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)

            rets[i] = torch.from_numpy(p_t)


        return rets



