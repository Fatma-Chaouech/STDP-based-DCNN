import torch
import torchvision
from data.preprocess import S1Transform
from models import utils
from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from predict import pass_through_network, eval
import logging
from models.model import Network


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test(dataset, device, model_directory, weights_name, classifier_name, data_directory, args):
    pt_path = model_directory + classifier_name
    net_path = model_directory + weights_name
    kernels = [utils.DoGKernel(7, 1, 2),
               utils.DoGKernel(7, 2, 1)]
    filter = utils.Filter(kernels, padding=3, threshold=50)
    s1_transform = S1Transform(filter)
    loader = get_loader(dataset, data_directory, s1_transform)
    model = Network()
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    clf = torch.load(pt_path, map_location=device)

    test_X, test_y = pass_through_network(
        model=model, loader=loader, device=device)
    predictions = clf.predict(test_X)
    accuracy, error, silence = eval(test_X, test_y, predictions)
    logger.info(
        f'\n-------- Accuracy : {accuracy} --------\n-------- Error : {error} --------\n-------- Silence : {silence} --------')


def get_loader(dataset, data_directory, s1_transform):
    if dataset == 'MNIST':
        test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_directory,
                                                             train=False, download=True,
                                                             transform=s1_transform))
    else:
        test = datasets.ImageFolder(root=dataset, transform=s1_transform)
    return DataLoader(test, batch_size=len(test), shuffle=False)
