import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FashionProductDataset:
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0] # number of rows in the dataframe

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['imagePath']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255)) # make a new image

        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[idx] # the label given an index

        # break down labels into categories
        gender_label = label['gender']
        article_label = label['articleType']
        color_base_label = label['baseColour']
        season_label = label['season']
        usage_label = label['usage']

        return {
            'image': image,
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'articleType': torch.tensor(article_label, dtype=torch.long),
            'baseColour': torch.tensor(color_base_label, dtype=torch.long),
            'season': torch.tensor(season_label, dtype=torch.long),
            'usage': torch.tensor(usage_label, dtype=torch.long)
        }