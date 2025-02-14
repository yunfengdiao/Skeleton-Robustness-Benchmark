from BayesianTrainer.PDBATrainer import PDBATrainer

def loadTrainer(args):
    adTrainer = ''
    if args.adTrainer == 'PDBATrainer':
        adTrainer = PDBATrainer(args)

    else:
        print('No Trainer created')

    return adTrainer