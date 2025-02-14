import sys
sys.path.insert(0, '')
import numpy as np
from classifiers.ActionClassifier import ActionClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from classifiers.msg3d.model.msg3d import *
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from classifiers.utils import *
import copy

class MSG3D(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.creatModel()
        self.steps = [30, 40]
        
    def creatModel(self):

        class Classifier(nn.Module):
            def __init__(self, args, dataloader,
                         num_point =25,
                         num_gcn_scales=13,
                         num_g3d_scales=6):
                super().__init__()
                self.args = args
                self.parents = dataloader.dataset.parents
                in_channels = args.inputDim
                num_class = args.classNum
                if args.dataset == 'hdm05':
                    graph='classifiers.msg3d.graph.hdm05.AdjMatrixGraph'
                    num_person = 1
                elif args.dataset == 'ntu60' or args.dataset == 'ntu120' or args.dataset == 'ntu60_300':
                    graph='classifiers.msg3d.graph.ntu_rgb_d.AdjMatrixGraph'
                    num_person = 2
                Graph = import_class(graph)
                A_binary = Graph().A_binary

                self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
                
                c1 = 96
                c2 = c1 * 2     
                c3 = c2 * 2     

                self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
                self.sgcn1 = nn.Sequential(
                    MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
                    MS_TCN(c1, c1),
                    MS_TCN(c1, c1))
                self.sgcn1[-1].act = nn.Identity()
                self.tcn1 = MS_TCN(c1, c1)

                self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
                self.sgcn2 = nn.Sequential(
                    MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
                    MS_TCN(c1, c2, stride=2),
                    MS_TCN(c2, c2))
                self.sgcn2[-1].act = nn.Identity()
                self.tcn2 = MS_TCN(c2, c2)

                self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
                self.sgcn3 = nn.Sequential(
                    MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
                    MS_TCN(c2, c3, stride=2),
                    MS_TCN(c3, c3))
                self.sgcn3[-1].act = nn.Identity()
                self.tcn3 = MS_TCN(c3, c3)

                self.fc = nn.Linear(c3, num_class)

            def forward(self, x):
                if self.args.classifier == 'MSG3D_bone' or self.args.baseClassifier == 'MSG3D_bone':
                    x_bone = x - x[:,:,:,self.parents,:]
                    x = x_bone
                N, C, T, V, M = x.size()
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                x = self.data_bn(x)
                x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

                x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
                x = self.tcn1(x)

                x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
                x = self.tcn2(x)

                x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
                x = self.tcn3(x)

                out = x
                out_channels = out.size(1)
                out = out.view(N, M, out_channels, -1)
                out = out.mean(3)   
                out = out.mean(1)   

                out = self.fc(out)
                return out
       
        if self.args.routine == 'train' or  self.args.routine == 'adTrain' or  self.args.routine == 'DualBayesian' or self.args.routine == 'finetune' or self.args.routine == 'attack' \
                or self.args.routine == 'bayesianTrain':
            self.model = Classifier(self.args, self.trainloader)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(self.args, self.testloader)
            self.model.eval()
        else:
            print("no model is created")

        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'


        if len(self.args.trainedModelFile) > 0 :
            if len(self.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
            else:
                if self.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                        self.args.baseClassifier + '/' + self.args.trainedModelFile))
                else:
                    if len(self.args.initWeightFile) > 0:
                        self.model.load_state_dict(
                            torch.load(self.retFolder + self.args.initWeightFile))
                    else:
                        self.model.load_state_dict(
                            torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))
        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)
        
    def configureOptimiser(self):
        if self.args.optimiser == 'Adam':
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.learningRate, weight_decay=0.0005)
        elif self.args.optimiser == 'SGD':
            self.optimiser = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.learningRate,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=0.0005)

    def adjustLearningRate(self, epoch):
        if self.args.optimiser == 'SGD':
            if self.steps:
                lr = self.args.learningRate * (
                    0.1**np.sum(epoch >= np.array(self.steps)))
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            else:
                self.lr = self.args.learningRate

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self):
        self.model.train()
    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.model(X)
    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)
    def train(self):
        size = len(self.trainloader.dataset)

        bestLoss = np.infty
        bestValAcc = 0

        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            self.adjustLearningRate(ep)
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                pred = self.model(X)
                loss = self.classLoss(pred, y)
                
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                    
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.retFolder + 'minLossModel.pth')
                bestLoss = epLoss
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = torch.argmax(self.model(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)
            acc = 1 - misclassified / len(self.testloader.dataset)
            logger.add_scalar('Loss/testing accuracy', acc, ep)
            self.model.train()
            if acc > bestValAcc:
                print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                torch.save(self.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                bestValAcc = acc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime)/3600

    def finetune(self):
        source_model_path = self.retFolder + '/minValLossModel.pth'
        pre_model_dic = torch.load(source_model_path)
        self.model.load_state_dict(pre_model_dic)

        print('------------------------------------------------------------------------------')
        print("the pretrained model has been loaded, finetune starts")
        mean_model = copy.deepcopy(self.model)
        sqmean_model = copy.deepcopy(self.model)

        self.optimiser = optim.SGD(self.model.parameters(), lr=self.args.learningRate, momentum=0.9, weight_decay=5e-4)

        size = len(self.trainloader.dataset)
        bestLoss = np.infty


        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0

        n_ensembled = 0
        for epoch in range(self.args.epochs):
            bestValAcc = 0
            epLoss = 0
            batchNum = 0
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                self.model.train()
                pred = self.model(X)
                loss = nn.functional.cross_entropy(pred, y)

                # finetune
                self.optimiser.zero_grad()
                loss.backward()
                grad_normal = get_grad(self.model)
                norm_grad_normal = cat_grad(grad_normal).norm()  
                add_into_weights(self.model, grad_normal, gamma=+0.1 / (norm_grad_normal + 1e-20))
                loss_add = self.classLoss(self.model(X), y)
                self.optimiser.zero_grad()
                loss_add.backward()
                grad_add = get_grad(self.model)
                add_into_weights(self.model, grad_normal, gamma=-0.1 / (norm_grad_normal + 1e-20))
                self.optimiser.zero_grad()
                grad_new_dict = OrderedDict()
                for (name, g_normal), (_, g_add) in zip(grad_normal.items(), grad_add.items()):
                    grad_new_dict[name] = g_normal + (self.args.lam / 0.1) * (g_add - g_normal)
                assign_grad(self.model, grad_new_dict)
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {epoch}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

                if ((epoch + 1) > self.args.swa_start
                        and ((epoch - self.args.swa_start) * len(self.trainloader) + batch) % (
                                ((self.args.epochs - self.args.swa_start) * len(
                                    self.trainloader)) // self.args.swa_n) == 0):
                    update_swag_model(self.model, mean_model, sqmean_model, n_ensembled)
                    n_ensembled += 1

        
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, epoch)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {epoch} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                bestLoss = epLoss
            print(f"epoch: {epoch} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            torch.save({"state_dict": self.model.state_dict(),
                        "opt_state_dict": self.optimiser.state_dict(),
                        "epoch": epoch},
                       os.path.join(self.retFolder + 'ep.pt'))
            torch.save({"mean_state_dict": mean_model.state_dict(),
                        "sqmean_state_dict": sqmean_model.state_dict(),
                        "epoch": epoch},
                       os.path.join(self.retFolder + 'swag_ep.pt'))

            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = torch.argmax(self.model(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)
            acc = 1 - misclassified / len(self.testloader.dataset)
            logger.add_scalar('Loss/testing accuracy', acc, epoch)
            self.model.train()
            if acc > bestValAcc:
                print(f"epoch: {epoch} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")

                torch.save(self.model.state_dict(), self.retFolder + 'finetune.pth')

                bestValAcc = acc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime) / 3600

    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            pred = torch.argmax(self.model(tx), dim=1)
            results[v*self.args.batchSize:(v+1)*self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)

        acc = 1 - misclassified / len(self.testloader.dataset)
        foolrate = misclassified / len(self.testloader.dataset)

        print(f"foolrate: {foolrate:>4f}")
        print(f"accuracy: {acc:>4f}")


        return acc

    def collectCorrectPredictions(self):
        self.model.eval()
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder+self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)

class MSG3D_ensemble(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'
        self.retFolder_joint = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '_joint' + '/'
        self.retFolder_bone = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '_bone' + '/'
        self.creatModel()

    def creatModel(self):
        class Classifier(nn.Module):
            def __init__(self, args, dataloader,
                         num_point =25,
                         num_gcn_scales=13,
                         num_g3d_scales=6):
                super().__init__()
                self.args = args
                self.parents = dataloader.dataset.parents
                self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'
                self.retFolder_joint = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '_joint' + '/'
                self.retFolder_bone = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '_bone' + '/'
                in_channels = args.inputDim
                num_class = args.classNum

                if args.dataset == 'hdm05':
                    graph='classifiers.msg3d.graph.hdm05.AdjMatrixGraph'
                    num_person = 1
                elif args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    graph='classifiers.msg3d.graph.ntu_rgb_d.AdjMatrixGraph'
                    num_person = 2
                self.model_joint = Model(num_class=num_class,num_point=num_point,
                                        num_person=num_person,num_gcn_scales=num_gcn_scales,
                                        num_g3d_scales=num_g3d_scales,graph=graph)
                self.model_bone = Model(num_class=num_class,num_point=num_point,
                                        num_person=num_person,num_gcn_scales=num_gcn_scales,
                                        num_g3d_scales=num_g3d_scales,graph=graph)

                self.model_bone.to(device)
                self.model_joint.to(device)
                if len(self.args.trainedModelFile) > 0 or len(self.args.initWeightFile) > 0:

                    if self.args.flag1==True and self.args.flag2==False:
                        self.model_joint.load_state_dict((torch.load(self.retFolder_joint + self.args.trainedModelFile))["mean_state_dict"])
                        self.model_bone.load_state_dict((torch.load(self.retFolder_joint + self.args.trainedModelFile))["mean_state_dict"])

                    if self.args.flag1 == True and self.args.flag2 == True:
                        self.model_joint.load_state_dict(
                            (torch.load(self.retFolder_joint + self.args.trainedModelFile))["sqmean_state_dict"])
                        self.model_bone.load_state_dict(
                            (torch.load(self.retFolder_joint + self.args.trainedModelFile))["sqmean_state_dict"])

                    if self.args.flag1==False and self.args.flag2==False:
                        if len(self.args.adTrainer) == 0:
                            self.model_joint.load_state_dict(torch.load(self.retFolder_joint + self.args.trainedModelFile))
                            self.model_bone.load_state_dict(torch.load(self.retFolder_bone + self.args.trainedModelFile))
                        if self.args.routine == 'bayesianTest':
                            print('----------------')
                            #self.retFolder=self.args.retPath + self.args.dataset + '/' + self.args.baseClassifier + '/'
                            self.retFolder=self.args.retPath
                            print( self.retFolder)
                            self.args.trainedModelFile='minValLossModel.pth'
                            #self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile,map_location='cuda:0'))
                            self.model_joint.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                                    self.args.baseClassifier + '_joint/' + self.args.trainedModelFile))
                            self.model_bone.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                                    self.args.baseClassifier + '_bone/' + self.args.trainedModelFile))
                        else:
                            if self.args.bayesianTraining:
                                self.model_joint.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                                    self.args.baseClassifier + '_joint/' + self.args.trainedModelFile))
                                self.model_bone.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                                    self.args.baseClassifier + '_bone/' + self.args.trainedModelFile))
                            else:
                                    a = torch.load(self.retFolder_joint + self.args.adTrainer + '/' + self.args.trainedModelFile)
                                    b = torch.load(self.retFolder_bone + self.args.adTrainer + '/' + self.args.trainedModelFile)
                                    self.model_joint.load_state_dict(torch.load(self.retFolder_joint + self.args.adTrainer + '/' + self.args.trainedModelFile))

                                    self.model_bone.load_state_dict(torch.load(self.retFolder_bone + self.args.adTrainer + '/' + self.args.trainedModelFile))

            def forward(self,x):
                x_bone = x - x[:,:,:,self.parents,:]
                out1 = self.model_joint(x)
                out2 = self.model_bone(x_bone)
                return out1+out2
        if self.args.routine == 'train' or self.args.routine == 'attack' \
                or self.args.routine == 'bayesianTrain':
            self.model = Classifier(self.args, self.trainloader,)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(self.args, self.testloader)
            self.model.eval()
        else:
            print("no model is created")
        self.model.to(device)
    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.model(X)

    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            pred = torch.argmax(self.model(tx), dim=1)
            results[v*self.args.batchSize:(v+1)*self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)

        acc = 1 - misclassified / len(self.testloader.dataset)
        foolrate = misclassified / len(self.testloader.dataset)

        print(f"foolrate: {foolrate:>4f}")
        print(f"accuracy: {acc:>4f}")

        return acc

    def collectCorrectPredictions(self):
        self.model.eval()

        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder+self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)