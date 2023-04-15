from torchvision import transforms
from models import utils, functional as sf
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class S1Transform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(
            timesteps, to_spike=True)
        self.cnt = 1

    def __call__(self, image):
        if self.cnt % 10000 == 0:
            logging.info(f'Preprocessed {self.cnt} images')
        self.cnt += 1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.byte()
