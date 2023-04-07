import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from .utils import to_pair

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.05):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = to_pair(kernel_size)
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight.requires_grad_(False)
        self.reset_weight(weight_mean, weight_std)

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        self.weight.normal_(weight_mean, weight_std)
    
    def forward(self, input):
        return fn.conv2d(input, self.weight)


class pooling(nn.Module):
    def __init__(self, kernel_size):
        super(Pooling, self).__init__()
        self.kernel_size = to_pair(kernel_size)
    
    def forward(self, input):
        return fn.max_pool2d(input, kernel_size, stride, padding)


class STDP(nn.Module):
    def __init__(self, conv_layer, learning_rate = (0.004, -0.003), use_stabilizer = True, lower_bound = 0, upper_bound = 1):
        super(STDP, self).__init__()
        self.conv_layer = conv_layer
        self.learning_rate = (Parameter(torch.tensor([learning_rate[0]])),
                            Parameter(torch.tensor([learning_rate[1]])))
        self.learning_rate[0].requires_grad_(False)
        self.learning_rate[1].requires_grad_(False)
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
        r"""Computes the ordering of the input and output spikes with respect to the position of each winner and
        returns them as a list of boolean tensors. True for pre-then-post (or concurrency) and False for post-then-pre.
        Input and output tensors must be spike-waves.

        Args:
            input_spikes (Tensor): Input spike-wave
            output_spikes (Tensor): Output spike-wave
            winners (List of Tuples): List of winners. Each tuple denotes a winner in a form of a triplet (feature, row, column).

        Returns:
            List: pre-post ordering of spikes
        """
        # accumulating input and output spikes to get latencies
        input_latencies = torch.sum(input_spikes, dim=0)
        output_latencies = torch.sum(output_spikes, dim=0)
        result = []
        for winner in winners:
            # generating repeated output tensor with the same size of the receptive field
            out_tensor = torch.ones(*self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
            # slicing input tensor with the same size of the receptive field centered around winner
            # since there is no padding, there is no need to shift it to the center
            in_tensor = input_latencies[:,winner[-2]:winner[-2]+self.conv_layer.kernel_size[-2],winner[-1]:winner[-1]+self.conv_layer.kernel_size[-1]]
            result.append(torch.ge(in_tensor,out_tensor))
        return result

    def forward(self, input_spikes, potentials, output_spikes, kwta = 3, inhibition_radius = 0):
        winners = sf.get_k_winners(potentials, kwta, inhibition_radius, output_spikes)
        pairings = self.get_pre_post_ordering(input_spikes, output_spikes, winners)
        lr = torch.zeros_like(self.conv_layer.weight)
        for i in range(len(winners)):
            winner = winners[i][0]
            lr[winner] = torch.where(pairings[i], *(self.learning_rate))
        self.conv_layer.weight += lr * ((self.conv_layer.weight-self.lower_bound) * (self.upper_bound-self.conv_layer.weight) if self.use_stabilizer else 1)
        self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)
    
    def update_learning_rate(self, ap, an):
        self.learning_rate[0] = ap
        self.learning_rate[1] = an