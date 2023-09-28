import os
import argparse
import utils.argument as argument
from utils.utils import setup_seed
from mainstay_training import mainstay_training


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)

    # arguments for defense
    parser = argument.add_defense_arguments(parser)

    # arguments for dataset
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number of images in one batch')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(seed=1)
    
    args = parser_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("Defense Method: {}".format(args.defense_method))
    if args.defense_method == 'mainstay':
        mainstay_training(args)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.defense_method))
