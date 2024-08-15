import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import io, transform
from torchvision import models, datasets, transforms
from alive_progress import alive_bar

class ISIC_Age_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['image'])
        self.df = self.df.set_index("image")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx] + '.jpg'
            if str(X) is not None:
                image = Image.open('/home/chenchen/dataset/ISIC_2019_Training_Input/%s' % X)
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                
                if(self.df["age_approx".strip()].iloc[idx].astype('int') > 59 or self.df["age_approx".strip()].iloc[idx].astype('int') < 40):
                    subg[0] = 1
                    subg[1] = 0
                else:
                    subg[0] = 0
                    subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg, X)
