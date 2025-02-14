import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class adv_dataset_hdm05(Dataset):
    def __init__(self, args):
        data_adv = torch.load(args.transfer_path)
        self.data_adv = data_adv[0]
        self.rlabels = data_adv[1]
        self.args = args
        self.data = self.data_adv
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate(
                (self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def __len__(self):
        return len(self.rlabels)

    def __getitem__(self, index):
        image_adv = self.data_adv[index]
        label = self.rlabels[index]
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            data_tensor = torch.tensor(image_adv, dtype=torch.float)
            C, T, V, M = data_tensor.shape

            if T != 64:
                t = torch.zeros([C, 64 - T, V, M], dtype=torch.float)
                data_tensor = torch.cat([data_tensor, t], dim=1)
            data_numpy = data_tensor.numpy()
            data_numpy = data_numpy[:, :, self.new_idx]
            return torch.from_numpy(data_numpy).to(device), label.to(device)
        return image_adv, label

    parents = np.array([10, 0, 1, 2, 3,
                        10, 5, 6, 7, 8,
                        10, 10, 11, 12, 13,
                        13, 15, 16, 17, 18,
                        13, 20, 21, 22, 23])

class adv_dataset_ntu(adv_dataset_hdm05):
    def __init__(self, args):
        data_adv = torch.load(args.transfer_path)
        self.data_adv = data_adv[0]
        self.rlabels = data_adv[1]
        self.args = args
        self.data = self.data_adv
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate(
                (self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)
    def __len__(self):
        return len(self.rlabels)

    def __getitem__(self, index):
        image_adv = self.data_adv[index]
        label = self.rlabels[index]
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            data_tensor = torch.tensor(image_adv, dtype=torch.float)
            C, T, V, M = data_tensor.shape

            if T != 64:
                t = torch.zeros([C, 64 - T, V, M], dtype=torch.float)
                data_tensor = torch.cat([data_tensor, t], dim=1)
            data_numpy = data_tensor.numpy()
            data_numpy = data_numpy[:, :, self.new_idx]
            return torch.from_numpy(data_numpy).to(device), label.to(device)
        return image_adv, label

    parents = np.array([1, 1, 21, 3, 21,
                        5, 6, 7, 21, 9,
                        10, 11, 1, 13, 14,
                        15, 1, 17, 18, 19,
                        2, 8, 8, 12, 12]) - 1

class CDataset(Dataset):
    def __init__(self, args, transform=None, target_transform=None):
        data = ''
        if args.routine == 'train'or  args.routine == 'adTrain' or  args.routine == 'DualBayesian' or args.routine == 'bayesianTrain'or args.routine == 'finetune':
            data = np.load(args.dataPath + '/' + args.dataset + '/' + args.trainFile)
        elif args.routine == 'attack' or args.routine == 'AR':
            if len(args.baseClassifier) > 0:
                data = np.load(args.retPath + '/' + args.dataset + '/' + args.classifier + '/'+ args.baseClassifier + '/' + args.adTrainer + '/' + args.trainFile)
            elif args.ensemble==True and ',' not in args.classifier:
                args.dataPath='../data/'
                args.trainFile='classTrain.npz'
                args.testFile='classTest.npz'
                data = np.load(args.dataPath + '/' + args.dataset + '/' + args.trainFile)
            else:
                data = np.load(args.retPath + '/' + args.dataset + '/' + args.classifier + '/' + args.trainFile)
        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or args.routine == 'bayesianTest':
            data = np.load(args.dataPath + '/' + args.dataset + '/' + args.testFile)
        else:
            print('Unknown routine, cannot create the dataset')
        self.data = data['clips']
        self.rlabels = data['classes']
        self.labels = torch.from_numpy(self.rlabels).type(torch.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.classNum = args.classNum
        self.args = args
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate(
                (self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        ##data format conversion to be consistent with ntu
        x = data.reshape((data.shape[0], -1, 3, 1))
        data = np.transpose(x, (2, 0, 1, 3))
        label = self.labels[idx]
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            data_tensor = torch.tensor(data, dtype=torch.float)
            C, T, V, M = data_tensor.shape

            if T != 64:
                t = torch.zeros([C, 64 - T, V, M], dtype=torch.float)
                data_tensor = torch.cat([data_tensor, t], dim=1)
            data_numpy = data_tensor.numpy()
            data_numpy = data_numpy[:, :, self.new_idx]
            return torch.from_numpy(data_numpy).float().to(device), label.to(device)
        return torch.from_numpy(data).float().to(device), label.to(device)
        # data = torch.from_numpy(data).float()
        # return data.to(device), label.to(device)

    # this is the topology of the skeleton, assuming the joints are stored in an array and the indices below
    # indicate their parent node indices. E.g. the parent of the first node is 10 and node[10] is the root node
    # of the skeleton
    parents = np.array([10, 0, 1, 2, 3,
                        10, 5, 6, 7, 8,
                        10, 10, 11, 12, 13,
                        13, 15, 16, 17, 18,
                        13, 20, 21, 22, 23])


class NTUDataset(CDataset):
    def __init__(self, args, transform=None, target_transform=None):
        super().__init__(args, transform, target_transform)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.args.classifier=='SkateFormer' or self.args.baseClassifier=='SkateFormer':
            data_tensor = torch.tensor(data, dtype=torch.float)
            C, T, V, M = data_tensor.shape

            if T != 64:
                t = torch.zeros([C, 64 - T, V, M], dtype=torch.float)
                data_tensor = torch.cat([data_tensor, t], dim=1)
            data_numpy = data_tensor.numpy()
            data_numpy = data_numpy[:, :, self.new_idx]
            return torch.from_numpy(data_numpy).float().to(device), label.to(device)
        return torch.from_numpy(data).to(device), label.to(device)

    # this is the topology of the skeleton, assuming the joints are stored in an array and the indices below
    # indicate their parent node indices.
    parents = np.array([1, 1, 21, 3, 21,
                        5, 6, 7, 21, 9,
                        10, 11, 1, 13, 14,
                        15, 1, 17, 18, 19,
                        2, 8, 8, 12, 12]) - 1


