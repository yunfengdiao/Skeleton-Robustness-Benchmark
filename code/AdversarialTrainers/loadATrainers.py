from AdversarialTrainers.EBMATrainer import EBMATrainer
from AdversarialTrainers.TRADES import TRADES
def loadATrainer(args):
    adTrainer = ''
    if args.adTrainer == 'EBMATrainer':
        adTrainer = EBMATrainer(args)
    elif args.adTrainer == 'TRADES':
        adTrainer = TRADES(args)
    else:
        print('No adTrainer created')

    return adTrainer