import torch.nn.functional as fn

def fire(potentials, threshold, return_thresholded_potentials=False):
    thresholded = potentials.clone().detach()
    fn.threshold_(thresholded, threshold, 0)
    if return_thresholded_potentials:
        return thresholded.sign(), thresholded
    return thresholded.sign()


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