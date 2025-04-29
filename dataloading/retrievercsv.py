import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

import dataloading.decoders
from torch.utils.data import Dataset
from tools.jpeg_utils import *
from tools.stego_utils import *

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class TrainCSVRetriever(Dataset):

    def __init__(self, data_path, dataframe, decoder='y', transforms=None, return_name=False):
        super().__init__()
        
        self.data_path = data_path
        self.dataframe = dataframe
        self.transforms = transforms
        self.decoder = decoder
        self.decode = getattr(dataloading.decoders, f'{decoder}_decode')    # get fn ptr for decoder
        self.labels = dataframe.label.values
        self.return_name = return_name

    def __getitem__(self, index: int):
        filename, label = self.dataframe.iloc[index][['filename', 'label']].tolist()
        image = self.decode(f'{self.data_path}/{filename}')

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
                    
        if self.return_name:
            return image, label, torch.as_tensor(dataloading.decoders.encode_string(filename))
        return image, label

    def __len__(self) -> int:
        return len(self.dataframe)

    def get_labels(self):
        return list(self.labels)