# STDP-based Spiking Deep Convolutional Neural Networks

This repository is a reimplementation of the paper [*STDP-based Spiking Deep Convolutional Neural Networks for Object Recognition* ](https://https://arxiv.org/abs/1611.01421) by Kheradpisheh et al (2017) using PyTorch.

The code is based on the original implementation in [SpykeTorch](https://github.com/miladmozafari/SpykeTorch), but has been simplified and streamlined for ease of use and readability. The main goal of this reimplementation is to provide a more focused and accessible implementation of the STDP-based spiking deep convolutional neural network architecture described in the paper.
## Getting Started

To get started, clone this repository and install the necessary dependencies:
```
git clone https://github.com/Fatma-Chaouech/STDP-based-DCNN.git
cd STDP-based-DCNN
pip install -r requirements.txt
```

Next, you can run the main training script to train the network on a dataset of your choice:
`python3 main.py --phase train --dataset <path-to-train-dataset>`

By default, the script will train the network on the MNIST dataset. You can specify a different dataset by providing the appropriate command line argument.

To test your model, you can run the following command in your terminal:` python3 --phase test --dataset <path-to-test-dataset> --weights_path <path-to-weights> --classifier_path <path-to-classifier>`
You can customize the values of --dataset, --weights_path, and --classifier_path to suit your specific needs. Additionally, you can edit these parameters in the [config.json](https://github.com/Fatma-Chaouech/STDP-based-DCNN/blob/main/configs/config.json) file.

For a detailed summary of the paper, please refer to the file [paper-summary.md](paper-summary.md).
