import torch
import torch.nn.functional as fn
import numpy as np
import math
import os


def to_pair(data):
    if isinstance(data, tuple):
        return data[0:2]
    return (data, data)


class FilterKernel:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass


class DoGKernel(FilterKernel):
    def __init__(self, window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self):
        w = self.window_size // 2
        x, y = np.mgrid[-w: w+1: 1, -w: w+1: 1]
        a = 1.0 / (2 * math.pi)
        prod = x * x + y * y
        f1 = (1 / (self.sigma1 * self.sigma1)) * \
            np.exp(-0.5 * (1 / (self.sigma1 * self.sigma1)) * prod)
        f2 = (1 / (self.sigma2 * self.sigma2)) * \
            np.exp(-0.5 * (1 / (self.sigma2 * self.sigma2)) * prod)
        dog = a * (f1 - f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()


class Filter:
    def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
        self.max_window_size = filter_kernels[0].window_size
        self.kernels = torch.stack([kernel().unsqueeze(0)
                                   for kernel in filter_kernels])
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        self.thresholds = thresholds
        self.use_abs = use_abs

    def __call__(self, input):
        output = fn.conv2d(input, self.kernels, padding=self.padding).float()
        output = torch.where(output < self.thresholds, torch.tensor(
            0.0, device=output.device), output)
        if self.use_abs:
            torch.abs_(output)
        return output


class Intensity2Latency:
    def __init__(self, timesteps=30, to_spike=False):
        self.timesteps = timesteps
        self.to_spike = to_spike

    def transform(self, intensities):
        bins_intensities = []
        nonzero_cnt = torch.nonzero(intensities).size()[0]
        bin_size = nonzero_cnt // self.timesteps
        intensities_flattened = torch.reshape(intensities, (-1,))
        intensities_flattened_sorted = torch.sort(
            intensities_flattened, descending=True)
        sorted_bins_value, sorted_bins_idx = torch.split(
            intensities_flattened_sorted[0], bin_size), torch.split(intensities_flattened_sorted[1], bin_size)
        spike_map = torch.zeros_like(intensities_flattened_sorted[0])
        for i in range(self.timesteps):
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
            spike_map_copy = spike_map.clone().detach()
            spike_map_copy = spike_map_copy.reshape(tuple(intensities.shape))
            bins_intensities.append(spike_map_copy.squeeze(0).float())
        return torch.stack(bins_intensities)

    def __call__(self, image):
        if self.to_spike:
            return self.transform(image).sign()
        return self.transform(image)


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")
                torch.save(target, save_path + ".cl")
                self.cache[index] = save_path
        else:
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    def reset_cache(self):
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)
