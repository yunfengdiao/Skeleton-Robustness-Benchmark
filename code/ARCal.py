# import os
# os.environ['R_HOME'] = 'Enter the R installation path here' 
import argparse
import warnings
from datasets.CDataset import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data
import torch.utils.data.distributed
import random
from joblib import Parallel, delayed

import numpy as np
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rcal1(inputnp, num_of_frame, label, c, r, d, e,args):
    tvrge = importr("tvReg")  # library R package
    pandas2ri.activate()
    Num = 60 // num_of_frame[r] + 1
    ## First body
    for i in range(25):
        for k in range(3):
            inputnp1 = inputnp[r, k, :num_of_frame[r], i, 0]
            inputnp1 = inputnp1.tolist()
            inputnp1 = robjects.FloatVector(inputnp1)
            # AR2
            result = tvrge.tvAR(inputnp1, p=2, type="none", est="ll")
            coef = result.rx2("coefficients").T
            coef = np.squeeze(coef)
            coef0 = coef[0, :]

            coef0 = np.pad(coef0, (0, 2), 'constant', constant_values=(0, 0))
            coef0 = np.tile(coef0, Num)
            coef0 = coef0[:60]
            coef1 = coef[1, :]
            coef1 = np.pad(coef1, (0, 2), 'constant', constant_values=(0, 0))
            coef1 = np.tile(coef1, Num)
            coef1 = coef1[:60]
            # AR1
            result1 = tvrge.tvAR(inputnp1, p=1, type="none", est="ll")
            coefar1 = result1.rx2("coefficients").T
            coefar1 = np.squeeze(coefar1)
            coefar1 = np.pad(coefar1, (0, 1), 'constant', constant_values=(0, 0))
            coefar1 = np.tile(coefar1, Num)
            coefar1 = coefar1[:60]
            # replace the first parameters
            coef01 = coef[0, :]
            coef01 = np.pad(coef01, (1, 1), 'constant', constant_values=(0, 0))
            coef01[0] = coefar1[0]
            coef01 = np.tile(coef01, Num)
            coef01 = coef01[:60]

            c = c.copy()
            c[k, :, i, 0] = coef0
            d = d.copy()
            d[k, :, i, 0] = coef1
            e = e.copy()
            e[k, :, i, 0] = coef01
    if args.dataset=='ntu60'or args.dataset=='ntu120':
    ## Second body
        for i in range(25):
            for k in range(3):
                inputnp2 = inputnp[r, k, :num_of_frame[r], i, 1]
                inputnp2 = inputnp2.tolist()
                inputnp2 = robjects.FloatVector(inputnp2)
                result = tvrge.tvAR(inputnp2, p=2, type="none", est="ll")

                coef = result.rx2("coefficients").T
                coef = np.squeeze(coef)
                coef0 = coef[0, :]
                coef0 = np.pad(coef0, (0, 2), 'constant', constant_values=(0, 0))
                coef0 = np.tile(coef0, Num)
                coef0 = coef0[:60]
                coef1 = coef[1, :]
                coef1 = np.pad(coef1, (0, 2), 'constant', constant_values=(0, 0))
                coef1 = np.tile(coef1, Num)
                coef1 = coef1[:60]

                result1 = tvrge.tvAR(inputnp2, p=1, type="none", est="ll")
                coefar2 = result1.rx2("coefficients").T
                coefar2 = np.squeeze(coefar2)
                coefar2 = np.pad(coefar2, (0, 1), 'constant', constant_values=(0, 0))
                coefar2 = np.tile(coefar2, Num)
                coefar2 = coefar2[:60]

                coef02 = coef[0, :]
                coef02 = np.pad(coef02, (1, 1), 'constant', constant_values=(0, 0))
                coef02[0] = coefar2[0]
                coef02 = np.tile(coef02, Num)
                coef02 = coef02[:60]

                c[k, :, i, 1] = coef0
                d[k, :, i, 1] = coef1
                e[k, :, i, 1] = coef02
    return c, d, e

def autore(input, num_of_frame, label,args):
    inputnp = input.detach().cpu().numpy()
    num_of_frame = num_of_frame.detach().cpu().numpy()
    label =label.detach().cpu().numpy()
    if args.dataset == 'ntu60' or args.dataset == 'ntu120':
        camera = 2
    elif args.dataset == 'hdm05':
        camera = 1
    c = np.zeros((3, 60, 25, camera))
    d = np.zeros((3, 60, 25, camera))
    e = np.zeros((3, 60, 25, camera))
    # Parallel
    results= Parallel(n_jobs=8)(
        delayed(rcal1)(inputnp, num_of_frame, label, c, r, d, e,args)
        for r in range(8))
    results = np.asarray(results)
    c = results[:, 0, :, :, :, :].squeeze()
    d = results[:, 1, :, :, :, :].squeeze()
    e = results[:, 2, :, :, :, :].squeeze()
    c = e*c+d
    c[np.isnan(c)] = 0
    e[np.isnan(e)] = 0
    return c, e

def  ARcalculate(data_eval,args):
    save_path = args.retPath + args.dataset + '/' + args.classifier + '/' + args.baseClassifier + '/' + args.adTrainer + '/' + 'ARmodel' + '/'
    for ith, (ith_data, label) in enumerate(data_eval):
        save_path_AR=save_path
        input_tensor = ith_data.to(device)
        length = input_tensor.size(0)
        num_frames = input_tensor.size(2)
        frames = torch.full((length,), num_frames)
        input = input_tensor.clone()
        vel, vel1 = autore(input, frames, label,args)
        if not os.path.exists(save_path_AR):
            os.makedirs(save_path_AR)
        else:
            print(f"already exist")
        save_path_AR = save_path_AR + '%d.npz' % (ith)
        np.savez_compressed(save_path_AR, AR2=vel, AR1=vel1)
    return input

def mainAR(args):
    cudnn.deterministic = True
    ngpus_per_node = torch.cuda.device_count()
    main_worker(0, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == 'ntu60' or args.dataset == 'ntu120':

        dataset = NTUDataset(args)
    elif args.dataset == 'hdm05':

        dataset = CDataset(args)

    cudnn.benchmark = True

    ## Data loading code

    loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    results = ARcalculate(loader,args)


