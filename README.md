# STDP-based Spiking Deep Convolutional Neural Networks

This repository is a reimplementation of the paper "STDP-based Spiking Deep Convolutional Neural Networks for Object Recognition" by Diehl and Cook (2015) using PyTorch.

The code is based on the original implementation in SpykeTorch, but has been simplified and streamlined for ease of use and readability. The main goal of this reimplementation is to provide a more focused and accessible implementation of the STDP-based spiking deep convolutional neural network architecture described in the paper.
## Getting Started

To get started, clone this repository and install the necessary dependencies:
'''
git clone https://github.com/Fatma-Chaouech/STDP-based-DCNN.git
cd STDP-based-DCNN
pip install -r requirements.txt
'''
Next, you can run the main training script to train the network on a dataset of your choice:
'''
python train.py --dataset <dataset-name>
'''
By default, the script will train the network on the MNIST dataset. You can specify a different dataset by providing the appropriate command line argument.
For a detailed summary of the paper, please refer to the file [paper-summary.md](paper-summary.md).
