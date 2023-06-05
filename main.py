import os

#Manually fix the GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

#Add EDM to path to load models properly
#   change this line to point to your specific path
sys.path.append("/home/sravula/MRI_Sampling_Diffusion/edm")

from utils.exp_utils import set_all_seeds, parse_args, parse_config
from learners.gradientlearner import MaskLearner

def main():
    args = parse_args(globals()['__doc__'])
    hparams = parse_config(args.config)

    print("\nWriting to ", os.path.join(hparams.save_dir, args.doc), '\n')

    set_all_seeds(hparams.seed)

    learner = MaskLearner(hparams, args)

    if args.test or args.baseline:
        learner.test()
    else:
        learner.run_meta_opt()

    return 0

if __name__ == '__main__':
    sys.exit(main())
