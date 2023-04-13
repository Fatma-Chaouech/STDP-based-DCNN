import torch
import torchvision
from data.preprocess import S1Transform
from models import utils
from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from predict import pass_through_network
import logging
from models.model import STDP


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test(dataset, device, model_path, model_name, data_path, args):
    pt_path = model_path + model_name
    net_path = model_path + "saved_l2.net"
    kernels = [utils.DoGKernel(7, 1, 2),
               utils.DoGKernel(7, 2, 1)]
    filter = utils.Filter(kernels, padding=3, thresholds=50)
    s1_transform = S1Transform(filter)
    loader = get_loader(dataset, data_path, s1_transform, args.batch_size)
    stdp = STDP()
    stdp.load_state_dict(torch.load(net_path))
    stdp.to(device)
    clf = torch.load(pt_path)

    test_X, test_y = pass_through_network(
        model=stdp, loader=loader, device=device)
    predictions = clf.predict(test_X, map_location=device)
    accuracy, error, silence = eval(test_X, test_y, predictions)
    logger.info(
        f'-------- Accuracy : {accuracy} --------\n-------- Error : {error} --------\n-------- Silence : {silence} --------')


def get_loader(dataset, data_path, s1_transform, batch_size=32):
    if dataset == 'MNIST':
        test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_path,
                                                             train=False, download=True,
                                                             transform=s1_transform))
    else:
        test = datasets.ImageFolder(root=dataset, transform=s1_transform)
    return DataLoader(test, batch_size=batch_size, shuffle=False)


def eval(X, y, predictions):
    non_silence_mask = np.count_nonzero(X, axis=1) > 0
    correct_mask = predictions == y
    correct_non_silence = np.logical_and(correct_mask, non_silence_mask)
    correct = np.count_nonzero(correct_non_silence)
    silence = np.count_nonzero(~non_silence_mask)
    return (correct / len(X), (len(X) - (correct + silence)) / len(X), silence / len(X))
