import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from SpikeTorch import snn
from SpikeTorch import functional as sf
from SpikeTorch import utils
from torchvision import transforms

use_cuda = True
device = torch.device('cuda')

class S1Transform:
    def __init__(self, filter, timesteps = 30):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps, to_spike=True)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt += 1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        if self.cnt == 1:
            print('Before DoG filter Shape :' + str(image.shape))
        image = self.filter(image)
        if self.cnt == 1:
            print('After DoG filter Shape :' + str(image.shape))
        image = sf.local_normalization(image, 8)
        if self.cnt == 1:
            print('After local normalization Shape :' + str(image.shape))
        temporal_image = self.temporal_transform(image)
        if self.cnt == 1:
            print('After temporal encoding Shape :' + str(temporal_image.shape))
        return temporal_image.byte()


class STDPMNIST(nn.Module):
    def __init__(self):
        super(STDPMNIST, self).__init__()

        self.conv1 = snn.Convolution(in_channels=2, out_channels=30, kernel_size=5)
        self.conv1_theshold = 15
        self.conv1_kwinners = 5
        self.conv1_inhibition_rad = 2

        self.conv2 = snn.Convolution(in_channels=30, out_channels=100, kernel_size=5)
        self.conv2_threshold = 10
        self.conv2_kwinners = 8
        self.conv2_inhibition_rad = 1

        self.stdp1 = snn.STDP(conv_layer=self.conv1)
        self.stdp2 = snn.STDP(conv_layer=self.conv2)

        # maximum value that + learning rate could take
        self.max_ap = torch.Tensor([0.15]).to(device)

        self.ctx = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        self.spk_cnt = 0


    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners
    

    def forward(self, input, layer_idx):
        input = sf.pad(input.float(), (2, 2, 2, 2))
        if self.training:
            potentials = self.conv1(input)
            spk, pot = sf.fire(potentials=potentials, threshold=self.conv1_theshold, return_thresholded_potentials=True)
            if layer_idx == 1:
                self.spk_cnt += 1
                if self.spk_cnt >= 500:
                    self.spk_cnt = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0]).to(device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.conv1_kwinners, self.conv1_inhibition_rad, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_pooling = sf.pooling(spk, 2, 2, 1)
            spk_in = sf.pad(spk_pooling, (1, 1, 1, 1))
            spk_in = sf.pointwise_inhibition(spk_in)
            potentials = self.conv2(spk_in)
            spk, pot = sf.fire(potentials, self.conv2_theshold, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.conv2_kwinners, self.conv2_inhibition_rad, spk)
            self.save_data(spk_in, pot, spk, winners)
            spk_out = sf.pooling(spk, spk.size()[2:])
            return spk_out
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_theshold, True)
            pooling = sf.pooling(spk, 2, 2, 1)
            padded = sf.pad(pooling, (1, 1, 1, 1))
            pot = self.conv2(padded)
            spk, pot = sf.fire(pot, self.conv2_threshold, True)
            spk = sf.pooling(spk, spk.size()[2:])
            print('Output shape : ', spk.shape)
            return spk


    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"])
        else:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"])
        


def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if i == 0:
            print('Image Shape :' + str(data_in.shape))
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)


def test(network, data, target, layer_idx):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        output,_ = network(data_in, layer_idx).max(dim = 0)
        ans[i] = output.reshape(-1).cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)




kernels = [utils.DoGKernel(7,1,2),
           utils.DoGKernel(7,2,1)]
filter = utils.Filter(kernels, padding = 3, thresholds = 50)
s1 = S1Transform(filter)

data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1))
MNIST_loader = DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

stdpmnist = STDPMNIST()
if use_cuda:
    stdpmnist.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
    stdpmnist.load_state_dict(torch.load("saved_l1.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data,_ in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(stdpmnist, data, 1)
            print("Done!")
            iter += 1
    torch.save(stdpmnist.state_dict(), "saved_l1.net")


# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
    stdpmnist.load_state_dict(torch.load("saved_l2.net"))
else:
    for epoch in range(2):
        print("Epoch", epoch)
        iter = 0
        for data,_ in MNIST_loader:
            print("Iteration", iter)
            train_unsupervise(stdpmnist, data, 2)
            print("Done!")
            iter += 1
    torch.save(stdpmnist.state_dict(), "saved_l2.net")




# Classification
# Get train data
for data,target in MNIST_loader:
    train_X, train_y = test(stdpmnist, data, target, 2)
    

# Get test data
for data,target in MNIST_testLoader:
    test_X, test_y = test(stdpmnist, data, target, 2)

# SVM
from sklearn.svm import LinearSVC
clf = LinearSVC(C=2.4)
clf.fit(train_X, train_y)
predict_train = clf.predict(train_X)
predict_test = clf.predict(test_X)

def get_performance(X, y, predictions):
    correct = 0
    silence = 0
    for i in range(len(predictions)):
        if X[i].sum() == 0:
            silence += 1
        else:
            if predictions[i] == y[i]:
                correct += 1
    return (correct/len(X), (len(X)-(correct+silence))/len(X), silence/len(X))

print(get_performance(train_X, train_y, predict_train))
print(get_performance(test_X, test_y, predict_test))