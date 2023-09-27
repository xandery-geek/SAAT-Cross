import os
import torch
import argparse
import utils.argument as argument
from mainstay_attack import mainstay_attack
from utils.utils import setup_seed


torch.multiprocessing.set_sharing_strategy('file_system')


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)
    parser = argument.add_attack_arguments(parser)
    
    # arguments for defense
    parser.add_argument('--adv', dest='adv', action="store_true", default=False,
                        help='load model with adversarial training')
    parser = argument.add_defense_arguments(parser)

    # arguments for dataset
    parser.add_argument('--bs', dest='bs', type=int, default=128, help='number of images in one batch')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(seed=1)
    
    args = parser_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("Current Method: {}".format(args.attack_method))
    if args.attack_method == 'mainstay':
        mainstay_attack(args)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.attack_method))
