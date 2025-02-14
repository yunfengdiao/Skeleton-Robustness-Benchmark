import copy
from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
from torch import nn
from optimisers.optimisers import SGAdaHMC
from classifiers.STGCN import STGCN
from classifiers.MSG3D import MSG3D,MSG3D_ensemble
from classifiers.AGCN import AGCN,AGCN_ensemble
from classifiers.STTFormer import STTFormer
from classifiers.SkateFormer import SkateFormer
from classifiers.CTRGCN import CTRGCN
from classifiers.FRHEAD import FRHEAD
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from datasets.dataloaders import *



class ExtendedBayesianClassifier(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        args.bayesianTraining = True
        self.trainloader, self.testloader = createDataLoader(args)
        if args.baseClassifier == 'STGCN':
            self.classifier = STGCN(args)
        if args.baseClassifier == 'CTRGCN':
            self.classifier = CTRGCN(args)
        if args.baseClassifier == '3layerMLP':
            self.classifier = ThreeLayerMLP(args)
        if args.baseClassifier == 'FRHEAD':
            self.classifier = FRHEAD(args)
        elif args.baseClassifier == 'MSG3D_joint' or args.baseClassifier == 'MSG3D_bone':
            self.classifier = MSG3D(args)
        elif args.baseClassifier == 'AGCN_joint' or args.baseClassifier == 'AGCN_bone':
            self.classifier = AGCN(args)
        elif args.baseClassifier == 'STTFormer':
            self.classifier =STTFormer(args)
        elif args.baseClassifier == 'SkateFormer':
            self.classifier =SkateFormer(args)

        self.classifier.setEval()
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/' + self.args.adTrainer + '/'
        self.createModel()

    def createModel(self):
        class AppendedModel(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.ReLU()
                )
               
            def forward(self, x):
                x = torch.nn.ReLU()(self.classifier.model(x))
                logits = self.model(x)
                logits = x + logits
                return logits

        self.modelList = [AppendedModel(self.classifier) for i in range(self.args.bayesianModelNum)]
        self.mean_model_list=[]
        self.sqmean_model_list = []
        self.eq_model_list = []

        for model in self.modelList:
            model.to(device)
            model.train()

        if len(self.args.trainedAppendedModelFile) > 0:
            for i in range(self.args.bayesianModelNum):
                self.modelList[i].model.load_state_dict(
                         torch.load(self.retFolder + '%i_minValLossAppendedModel.pth' % (i),map_location='cuda:0'))
                mean_model = copy.deepcopy(self.modelList[i])
                sqmean_model = copy.deepcopy(self.modelList[i])
                eq_model=copy.deepcopy(self.modelList[i])
                if self.args.routine=='attack' or self.args.routine=='bayesianTest':
                    state_dict = torch.load(
                            self.retFolder + 'Dual_Bayesian/multi_swag_ep_%i.pt' % (i),
                            map_location='cuda:0')
                    mean_model.load_state_dict(state_dict["mean_state_dict"])
                    sqmean_model.load_state_dict(state_dict["sqmean_state_dict"])

                self.eq_model_list.append(eq_model)
                self.mean_model_list.append(mean_model)
                self.sqmean_model_list.append(sqmean_model)

        self.configureOptimiser()
        self.classificationLoss()
        
    def configureOptimiser(self):

        self.optimiserList = [SGAdaHMC(self.modelList[i].model.parameters(), config=dict(lr=self.args.learningRate, alpha=0, gamma=0.01, L=30, T=1e-5, tao=2, C=1))
                              for i in range(self.args.bayesianModelNum)]

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def modelEval(self, X, modelNo=-1):
        if modelNo == -1:
            pred = torch.zeros((len(X), self.classifier.args.classNum), dtype=torch.float).to(device)
            for model in self.modelList:
                pred += model(X)
        else:
            pred = self.modelList[modelNo](X)

        return pred

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self, modelNo = -1):
        if modelNo == -1:
            for model in self.modelList:
                model.train()
        else:
            self.modelList[modelNo].train()

    def setEval(self, modelNo = -1):
        if modelNo == -1:
            for model in self.modelList:
                model.train()
        else:
            self.modelList[modelNo].eval()
            
    def train(self):
        return
    
    def test(self):
        misclassified = 0
        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            predY = self.modelEval(tx)
            predY = torch.argmax(predY, dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = predY.cpu()
            diff = (predY - ty) != 0
            misclassified += torch.sum(diff)
        error = misclassified / len(self.classifier.testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        return

    def collectCorrectPredictions(self):

        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            pred = torch.argmax(self.modelEval(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.classifier.testloader.dataset.data[results.astype(bool)]
        adLabels = self.classifier.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")
        path = self.retFolder
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)

class ExtendedBayesianClassifier_ensemble(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        args.bayesianTraining = True
        self.trainloader, self.testloader = createDataLoader(args)
        if args.baseClassifier == 'MSG3D':
            self.classifier = MSG3D_ensemble(args)
        if args.baseClassifier == 'AGCN':
            self.classifier = AGCN_ensemble(args)
        self.classifier.model.eval()
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.adTrainer + '/'
        self.retFolder_joint = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + args.baseClassifier+'_joint/' + self.args.adTrainer + '/'
        self.retFolder_bone = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + args.baseClassifier+'_bone/' + self.args.adTrainer + '/'
        self.createModel()
        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)
    def remove_prefix_from_state_dict(self,state_dict, prefix='model.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace(prefix, '') if k.startswith(prefix) else k
            new_state_dict[new_key] = v

        return new_state_dict

    def createModel(self):
        class AppendedModel(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier
                if len(classifier.testloader):
                    self.parents = classifier.testloader.dataset.parents
                else:
                    self.parents = classifier.trainloader.dataset.parents
                self.model_joint = torch.nn.Sequential(
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.ReLU()
                )
                self.model_bone = torch.nn.Sequential(
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.ReLU()
                )
            def forward(self, x):
                x_bone = x - x[:,:,:,self.parents,:]
                x = torch.nn.ReLU()(self.classifier.model.model_joint(x))
                logits = self.model_joint(x)
                logits = x + logits
                x_bone = torch.nn.ReLU()(self.classifier.model.model_bone(x_bone))
                logits_bone = self.model_bone(x)
                logits_bone = x_bone + logits_bone
                return logits+logits_bone

        self.modelList = [AppendedModel(self.classifier) for i in range(self.args.bayesianModelNum)]

        self.mean_model_list = []
        self.sqmean_model_list = []


        for model in self.modelList:
            model.to(device)


        if len(self.args.trainedAppendedModelFile) > 0:
            for i in range(self.args.bayesianModelNum):
                self.modelList[i].model_joint.load_state_dict(
                         torch.load(self.retFolder_joint + '%i_minValLossAppendedModel.pth' % (i),map_location='cuda:0'))
                self.modelList[i].model_bone.load_state_dict(
                         torch.load(self.retFolder_bone + '%i_minValLossAppendedModel.pth' % (i),map_location='cuda:0'))

                mean_model = copy.deepcopy(self.modelList[i])
                sqmean_model = copy.deepcopy(self.modelList[i])

                if self.args.routine == 'attack':
                    #joint
                    state_dict_joint = torch.load(self.retFolder_joint+ 'Dual_Bayesian/multi_swag_ep_%i.pt' % (i),
                                            map_location='cuda:0')
                    joint_mean_state_dict=state_dict_joint["mean_state_dict"]
                    joint_mean_state_dict = self.remove_prefix_from_state_dict(joint_mean_state_dict, prefix='model.')
                    mean_model.model_joint.load_state_dict(joint_mean_state_dict)
                    
                    joint_sqmean_state_dict=state_dict_joint["sqmean_state_dict"]
                    joint_sqmean_state_dict = self.remove_prefix_from_state_dict(joint_sqmean_state_dict, prefix='model.')
                    sqmean_model.model_joint.load_state_dict(joint_sqmean_state_dict)
                    #bone
                    state_dict_bone = torch.load(
                        self.retFolder_bone + 'Dual_Bayesian/multi_swag_ep_%i.pt' % (i),
                        map_location='cuda:0')
                    bone_mean_state_dict=state_dict_bone["mean_state_dict"]
                    bone_mean_state_dict = self.remove_prefix_from_state_dict(bone_mean_state_dict, prefix='model.')
                    mean_model.model_bone.load_state_dict(bone_mean_state_dict)
                    
                    bone_sqmean_state_dict=state_dict_bone["sqmean_state_dict"]
                    bone_sqmean_state_dict = self.remove_prefix_from_state_dict(bone_sqmean_state_dict, prefix='model.')
                    sqmean_model.model_bone.load_state_dict(bone_sqmean_state_dict)

                    self.mean_model_list.append(mean_model)
                    self.sqmean_model_list.append(sqmean_model)
       
                else:
                    self.mean_model_list.append(mean_model)
                    self.sqmean_model_list.append(sqmean_model)


        self.classificationLoss()

    def classificationLoss(self):
        self.classLoss = torch.nn.CrossEntropyLoss()

    def modelEval(self, X, modelNo = -1):
        if modelNo == -1:
            pred = torch.zeros((len(X), self.classifier.args.classNum), dtype=torch.float).to(device)
            for model in self.modelList:
                pred += model(X)
        else:
            pred = self.modelList[modelNo](X)
        return pred

    def classificationLoss(self):
        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self):
        for model in self.modelList:
            model.model.train()

    def setEval(self):
        for model in self.modelList:
            model.model_joint.eval()
            model.model_bone.eval()

    def train(self):
        return
  
    def test(self):

        misclassified = 0
        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            tx = tx.to(device)
            ty = ty.to(device)
            predY = self.modelEval(tx)
            predY = torch.argmax(predY, dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = predY.cpu()
            diff = (predY - ty) != 0
            misclassified += torch.sum(diff)

        error = misclassified / len(self.classifier.testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        print(f"error: {error:>4f}")

        return
    def collectCorrectPredictions(self):
        misclassified = 0
        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            pred = torch.argmax(self.modelEval(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.classifier.testloader.dataset.data[results.astype(bool)]
        adLabels = self.classifier.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")
        path = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/'+self.args.baseClassifier + '/'+self.args.adTrainer+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + self.args.adTrainFile, clips=adData, classes=adLabels)
