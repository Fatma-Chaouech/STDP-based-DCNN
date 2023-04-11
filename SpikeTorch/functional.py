import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
from .utils import to_pair


def fire(potentials, threshold, return_thresholded_potentials=False):
    thresholded = potentials.clone().detach()
    fn.threshold_(thresholded, threshold, 0)
    if return_thresholded_potentials:
        return thresholded.sign(), thresholded
    return thresholded.sign()


def pad(input, pad, value=0):
    return fn.pad(input, pad, value=value)


def pooling(input, kernel_size, stride=None, padding=0):
    return fn.max_pool2d(input, kernel_size, stride, padding)

def local_normalization(input, normalization_radius, eps=1e-12):
    length = normalization_radius * 2 + 1
    kernel = torch.ones(1, 1, length, length, device=input.device).float() / ((length) ** 2)
    y = input.squeeze(0)
    y.unsqueeze_(1) 
    means = fn.conv2d(y, kernel, padding=normalization_radius) + eps 
    y = y / means 
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y


# only one neuron can fire at each position (for != feature maps)
def pointwise_inhibition(thresholded_potentials):
    # maximum of each position in each time step
    maximum = torch.max(thresholded_potentials, dim=1, keepdim=True)
    # compute signs for detection of the earliest spike
    clamp_pot = maximum[0].sign()
    # maximum of clamped values is the indices of the earliest spikes
    clamp_pot_max_1 = (clamp_pot.size(0) - clamp_pot.sum(dim = 0, keepdim=True)).long()
    clamp_pot_max_1.clamp_(0, clamp_pot.size(0) - 1)
    ## last timestep of each feature map
    clamp_pot_max_0 = clamp_pot[-1:,:,:,:]
    # finding winners (maximum potentials between early spikes) (indices of winners)
    winners = maximum[1].gather(0, clamp_pot_max_1)
    # generating inhibition coefficient
    coef = torch.zeros_like(thresholded_potentials[0]).unsqueeze_(0)
    coef.scatter_(1, winners, clamp_pot_max_0)
    # applying inhibition to potentials (broadcasting multiplication)
    return torch.mul(thresholded_potentials, coef)


def get_k_winners(potentials, kwta = 1, inhibition_radius = 0, spikes = None):
    r"""Finds at most :attr:`kwta` winners first based on the earliest spike time, then based on the maximum potential.
    It returns a list of winners, each in a tuple of form (feature, row, column).

    .. note::

        Winners are selected sequentially. Each winner inhibits surrounding neruons in a specific radius in all of the
        other feature maps. Note that only one winner can be selected from each feature map.

    Args:
        potentials (Tensor): The tensor of input potentials.
        kwta (int, optional): The number of winners. Default: 1
        inhibition_radius (int, optional): The radius of lateral inhibition. Default: 0
        spikes (Tensor, optional): Spike-wave corresponding to the input potentials. Default: None

    Returns:
        List: List of winners.
    """
    if spikes is None:
        spikes = potentials.sign()
    # finding earliest potentials for each position in each feature
    maximum = (spikes.size(0) - spikes.sum(dim = 0, keepdim=True)).long()
    maximum.clamp_(0,spikes.size(0)-1)
    values = potentials.gather(dim=0, index=maximum) # gathering values
    # propagating the earliest potential through the whole timesteps
    truncated_pot = spikes * values
    # summation with a high enough value (maximum of potential summation over timesteps) at spike positions
    v = truncated_pot.max() * potentials.size(0)
    truncated_pot.addcmul_(spikes,v)
    # summation over all timesteps
    total = truncated_pot.sum(dim=0,keepdim=True)
    
    total.squeeze_(0)
    global_pooling_size = tuple(total.size())
    winners = []
    for k in range(kwta):
        max_val, max_idx = total.view(-1).max(0)
        if max_val.item() != 0:
            # finding the 3d position of the maximum value
            max_idx_unraveled = np.unravel_index(max_idx.item(), global_pooling_size)
            # adding to the winners list
            winners.append(max_idx_unraveled)
            # preventing the same feature to be the next winner
            total[max_idx_unraveled[0],:,:] = 0
            # columnar inhibition (increasing the chance of leanring diverse features)
            if inhibition_radius != 0:
                rowMin,rowMax = max(0,max_idx_unraveled[-2]-inhibition_radius),min(total.size(-2),max_idx_unraveled[-2]+inhibition_radius+1)
                colMin,colMax = max(0,max_idx_unraveled[-1]-inhibition_radius),min(total.size(-1),max_idx_unraveled[-1]+inhibition_radius+1)
                total[:,rowMin:rowMax,colMin:colMax] = 0
        else:
            break
    return winners
