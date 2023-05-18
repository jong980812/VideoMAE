import os
from statistics import NormalDist
import numpy as np
import torch
from torchvision import transforms
from util_tools.random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import util_tools.video_transforms as video_transforms 
import util_tools.volume_transforms as volume_transforms





def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
class Grad_sampling(Dataset):
    def __init__(self,
            crop_size=224, short_side_size=256, new_height=256,
            new_width=340, keep_aspect_ratio=True, num_segment=16,
            verb=None,noun=None,video_path='',
            args=None):
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.args = args
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")
        '''
        #! self.rand_erase = True -> 0.25
        '''
        self.dataset_samples = video_path
        verb_label_array = verb
        noun_label_array = noun
        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
                            
    def __getitem__(self, index):
        sample = self.dataset_samples
        buffer = self.loadvideo_decord(sample)   
        # buffer = self.data_transform(buffer)
        return buffer
    def __len__(self):
        return len(self.test_dataset)
    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []
        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index)) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

          