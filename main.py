import argparse
import torch
import os
import json

from train import train
from test import test


def main():
    config = get_config()
    device = get_device(config['use_cuda'])
    args = parse_args(config)
    if args.phase == 'train':
        train(args.dataset, device, config['model_path'],
              config['model_name'], config['data_path'], args)
    elif args.phase == 'test':
        test(args.dataset, device, config['model_path'],
             config['model_name'], config['data_path'], args)


def parse_args(config):
    pt_path = config['model_path'] + config['model_name']
    parser = argparse.ArgumentParser(
        description='Script to train or test the model.')
    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help='train or test phase')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training/testing')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--resume', default=None,
                        help='path to a saved model to resume training or test')
    parser.add_argument('--dataset', default='MNIST',
                        help='path to the dataset directory')
    parser.add_argument('--model_path', default=pt_path,
                        help='path to the pt file')
    args = parser.parse_args()
    return args


def get_device(use_cuda=True):
    if use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')


def get_config():
    with open('./configs/config.json') as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    main()
