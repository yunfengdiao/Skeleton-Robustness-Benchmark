from Attackers.SMART import SmartAttacker
from Attackers.I_FGSM import FGSMAttacker
from Attackers.MI_FGSM import MIAttacker
from Attackers.Augment import AugmentAttacker
from Attackers.BA import BayesianAttacker
from Attackers.MIG import MIGAttacker
from Attackers.CIASA import CIASAAttacker
from Attackers.ENSEMBLE import EnsembleAttacker
from Attackers.TASAR import TASARAttacker
from Attackers.SVRE import SVREAttacker

def loadAttacker(args):
    attacker = ''
    if args.attacker == 'SMART':
        attacker = SmartAttacker(args)
    if args.attacker == 'FGSM':
        attacker = FGSMAttacker(args)
    if args.attacker == 'MI-FGSM':
        attacker = MIAttacker(args)
    if args.attacker == 'Augment':
        attacker = AugmentAttacker(args)
    if args.attacker == 'MIG':
        attacker = MIGAttacker(args)
    if args.attacker == 'CIASA':
        attacker = CIASAAttacker(args)
    if args.attacker == 'BA':
        attacker = BayesianAttacker(args)
    if args.attacker == 'ENSEMBLE':
        attacker = EnsembleAttacker(args)
    if args.attacker == 'TASAR':
        attacker = TASARAttacker(args)
    if args.attacker == 'SVRE':
        attacker = SVREAttacker(args)

    else:
        print('No classifier created')
    return attacker