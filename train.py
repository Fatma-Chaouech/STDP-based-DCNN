import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from data.preprocess import S1Transform
from models import utils
import torch.utils.data as data
from sklearn.svm import LinearSVC
import torchvision.datasets as datasets
from models.model import STDP
import logging

from predict import pass_through_network

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train(dataset, device, model_directory, classifier_name, data_directory, args):
    kernels = [utils.DoGKernel(7, 1, 2),
               utils.DoGKernel(7, 2, 1)]
    filter = utils.Filter(kernels, padding=3, thresholds=50)
    s1_transform = S1Transform(filter)
    stdp = STDP(device).to(device)
    loader = get_loader(dataset, data_directory,
                        s1_transform, args.batch_size)
    train_layer(1, model=stdp, loader=loader,
                model_directory=model_directory, device=device)
    train_layer(2, model=stdp, loader=loader,
                model_directory=model_directory, device=device)
    train_classifier(stdp, loader, device, model_directory, classifier_name)


def get_loader(dataset, data_directory, s1_transform, batch_size=32):
    if dataset == 'MNIST':
        train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_directory,
                                                              train=True, download=True,
                                                              transform=s1_transform))
    else:
        train = datasets.ImageFolder(root=dataset, transform=s1_transform)
    return DataLoader(train, batch_size=batch_size, shuffle=False)


def train_layer(num_layer, model, loader, model_directory, device='cuda'):
    model.train()

    if num_layer == 1:
        name = 'first'
    else:
        name = 'second'
    net_path = model_directory + "saved_l" + str(num_layer) + ".net"

    logger.info("Training the " + name + " layer...")
    if os.path.isfile(net_path):
        model.load_state_dict(torch.load(net_path))
    else:
        learning_convergence = 1
        epoch = 1
        while learning_convergence >= 0.01:
            logger.info(f"============ Epoch {epoch} ============")
            iter = 0
            for data, _ in loader:
                logger.info("Iteration", iter)
                train_unsupervised(model, data, num_layer, device)
                iter += 1
            epoch += 1
            learning_convergence = calculate_learning_convergence()

        torch.save(model.state_dict(), net_path)


def train_unsupervised(model, data, layer_idx, device):
    for i in range(len(data)):
        data_in = data[i].to(device)
        model(data_in, layer_idx)
        model.stdp(layer_idx)


def train_classifier(model, loader, device, model_directory, classifier_name, C=2.4):
    logger.info('Training the classifier...')

    Xtrain_path = 'tmp/train_x.npy'
    ytrain_path = 'tmp/train_y.npy'
    pt_path = model_directory + classifier_name

    # setting the model on prediction mode
    model.eval()
    train_X, train_y = pass_through_network(
        model, loader, Xtrain_path, ytrain_path, device)

    clf = LinearSVC(C=C)
    clf.fit(train_X, train_y)
    torch.save(clf, pt_path)


def calculate_learning_convergence(weights):
    n_w = weights.size
    sum_wf_i = np.sum(weights * (1 - weights))
    c_l = sum_wf_i / n_w
    return c_l
