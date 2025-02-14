

class ClassifierArgs:
    def __init__(self, args):
        self.args = args
        self.dataFolder = args.dataPath
        self.retFolder = args.retPath
        self.trainFile = args.trainFile
        self.testFile = args.testFile
        self.trainedModelFile = args.trainedModelFile
        self.classNum = args.classNum
        self.batchSize = args.batchSize
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.routine = args.routine
        self.classifier = args.classifier
        self.adTrainFile = args.adTrainFile


class ActionClassifier:
    def __init__(self, args):
        self.args = args
        self.loss = ''
        self.model = ''


    def train(self):
        return

    def test(self):
        return

    def collectCorrectPredictions(self):
        return



