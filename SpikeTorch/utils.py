import torch.nn.functional as fn
import torch 

class FilterKernel:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass



class DoGKernel(FilterKernel):
    def __init__(self, , window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def __call__(self):
        w = self.window_size // 2
        x, y = np.mgrid[-w: w+1: 1, -w: w+1: 1]
        a = 1.0 / (2 * math.pi)
        prod = x * x + y * y
        f1 = (1 / (self.sigma1 * self.sigma1)) * np.exp(-0.5 * (1 / (self.sigma1 * self.sigma1)) * prod)
        f2 = (1 / (self.sigma2 * self.sigma2)) * np.exp(-0.5 * (1 / (self.sigma2 * self.sigma2)) * prod)
        dog = a * (f1 - f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()



class Filter:
    def __init__(self, filter_kernels, padding=0, thresholds=None):
        self.filter_kernels = filter_kernels
        self.max_window_size = filter_kernels[0].window_size
        self.kernels = torch.stack([kernel().unsqueeze(0) for kernel in filter_kernels])
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        self.thresholds = thresholds
        # self.use_abs = use_abs

    def __call__(self):
        output = fn.conv2d(input, self.kernels).float()
        output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
        # if self.use_abs:
        #     torch.abs_(output)
        return output



class Intensity2Latency:
    def __init__(self, timesteps, to_spike=False):
        self.timesteps = timesteps
        self.to_spike = to_spike
      

    def transform(self, intensities):
        bins_intencities = []
        nonzero_cnt = torch.nonzero(intencities).size()[0]

        bin_size = nonzero_cnt//self.time_steps

        intencities_flattened = torch.reshape(intencities, (-1,))
        intencities_flattened_sorted = torch.sort(intencities_flattened, descending=True)

        sorted_bins_value, sorted_bins_idx = torch.split(intencities_flattened_sorted[0], bin_size), torch.split(intencities_flattened_sorted[1], bin_size)

        spike_map = torch.zeros_like(intencities_flattened_sorted[0])
    
        for i in range(self.time_steps):
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
            spike_map_copy = spike_map.clone().detach()
            spike_map_copy = spike_map_copy.reshape(tuple(intencities.shape))
            bins_intencities.append(spike_map_copy.squeeze(0).float())
    
        return torch.stack(bins_intencities)
    
    def __call__(self):
        if self.to_spike:
            return self.intensity_to_latency(image).sign()
        return self.intensity_to_latency(image)


def to_pair(data):
    if isinstance(data, tuple):
        return data[0:2]
    return (data, data)