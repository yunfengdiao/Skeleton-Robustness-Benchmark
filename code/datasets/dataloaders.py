import random
from torch.utils.data import DataLoader
from datasets.CDataset import *
import numpy as np

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def createDataLoader(args):
    trainloader = ''
    testloader = ''
    if args.dataset == 'hdm05' or args.dataset == 'ntu60' or args.dataset == 'ntu120':
        if args.routine == 'train' or  args.routine == 'adTrain'or args.routine == 'DualBayesian' or args.routine == 'finetune' or args.routine == 'bayesianTrain':
            if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                traindataset = NTUDataset(args)
            else:
                traindataset = CDataset(args)

            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=True)

            if len(args.testFile):
                routine = args.routine
                args.routine = 'test'
                if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    testdataset = NTUDataset(args)
                else:
                    testdataset = CDataset(args)
                args.routine = routine
                testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False)

        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or args.routine == 'bayesianTest':
            if args.transfer_attack == True:
                if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    testdataset = adv_dataset_ntu(args)
                else:
                    testdataset = adv_dataset_hdm05(args)
            else:
                if len(args.testFile):
                    if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                        testdataset = NTUDataset(args)
                    else:
                        testdataset = CDataset(args)
            testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False)

        elif args.routine == 'attack':
            ##
            if args.transfer_attack == True:
                if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    testdataset = adv_dataset_ntu(args)
                else:
                    testdataset = adv_dataset_hdm05(args)
            else:
                if len(args.testFile):
                    if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                        testdataset = NTUDataset(args)
                    else:
                        testdataset = CDataset(args)
            testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False)
            if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                traindataset = NTUDataset(args)
            else:
                traindataset = CDataset(args)
            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=False, drop_last=True)
    else:
        print ('No dataset is loaded')

    return trainloader, testloader

