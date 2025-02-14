import os
from classifiers.ActionClassifier import ActionClassifier
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn
import torch.optim as optim
from datasets.dataloaders import *
from classifiers.sttformer.sttformer import Model
from classifiers.utils import *
import copy

class STTFormer(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()

        self.steps = [60, 80]

    def createModel(self):
        config = [[64, 64, 16], [64, 64, 16],
                  [64, 128, 32], [128, 128, 32],
                  [128, 256, 64], [256, 256, 64],
                  [256, 256, 64], [256, 256, 64]]

        if self.args.dataset == 'ntu60':
            Classifier = Model(num_classes=60, num_frames=60, config=config)
        elif self.args.dataset == 'ntu120':
            Classifier = Model(num_classes=120, num_frames=60, config=config)
        else:
            Classifier = Model(num_classes=65, num_frames=60, config=config)
        # create the train data loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train'or  self.args.routine == 'adTrain' or self.args.routine == 'finetune' or self.args.routine == 'attack' or self.args.routine == 'DualBayesian' \
                or self.args.routine == 'bayesianTrain':
            self.model = Classifier
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier
            self.model.eval()
        else:
            print("no model is created")

        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'

        if len(self.args.trainedModelFile) > 0 :
            if len(self.args.adTrainer) == 0:
                # self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
                self.model.load_state_dict(
                    torch.load(self.retFolder + self.args.trainedModelFile, map_location='cuda:0'))
            else:
                if self.args.bayesianTraining:
                    # self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                    #                     self.args.baseClassifier + '/' + self.args.trainedModelFile))
                    self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                                          self.args.baseClassifier + '/' + self.args.trainedModelFile,
                                                          map_location='cuda:0'))
                else:
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))

        # self.model = nn.DataParallel(self.model)
        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)

    def configureOptimiser(self):
        if self.args.optimiser == 'SGD':
            self.optimiser = optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0004)
        elif self.args.optimiser == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.1,
                weight_decay=0.0004)

    def adjust_learning_rate(self, epoch):
        if self.args.optimiser == 'SGD' or self.args.optimiser == 'Adam':
            if epoch < 5:
                lr = 0.1 * (epoch + 1) / 5
            else:
                lr = 0.1 * (0.1 ** np.sum(epoch >= np.array([self.steps])))
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def classificationLoss(self):
        self.classLoss = nn.CrossEntropyLoss()

    def setTrain(self):
        self.model.train()

    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo=-1):
        return self.model(X)

    # this function is to train the classifier from scratch
    def train(self):
        self.model.train()
        size = len(self.trainloader.dataset)
        bestLoss = np.infty
        bestValAcc = 0
        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0
        global_step = 0

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0

            # print(f"epoch: {ep} GPU memory allocated: {torch.cuda.memory_allocated(1)}")
            for batch, (X, y) in enumerate(self.trainloader):
                X, y = X.to(device), y.long().to(device)
                self.adjust_learning_rate(ep)
                global_step += 1
                batchNum += 1
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.classLoss(pred, y)

                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            # print(f"epoch: {ep} GPU memory allocated after one epoch: {torch.cuda.memory_allocated(1)}")
            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.retFolder + 'minLossModel.pth')
                bestLoss = epLoss
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            # run validation and save a model if the best validation loss so far has been achieved.
            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                tx, ty = tx.to(device), ty.long().to(device)
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
            valTime += (valEndTime - valStartTime) / 3600
            # print(f"epoch: {ep} GPU memory allocated after one epoch validation: {torch.cuda.memory_allocated(1)}")


    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            tx, ty = tx.to(device), ty.long().to(device)
            pred = torch.argmax(self.model(tx), dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)
        acc = 1 - misclassified / len(self.testloader.dataset)
        foolrate = misclassified / len(self.testloader.dataset)

        print(f"foolrate: {foolrate:>4f}")
        print(f"accuracy: {acc:>4f}")

        return acc

    # this function is to collected all the testing samples that can be correctly collected
    # by the pre-trained classifier, to make a dataset for adversarial attack
    def collectCorrectPredictions(self):
        self.model.eval()

        # collect data from the test data
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            tx, ty = tx.to(device), ty.long().to(device)
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder + self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)

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
