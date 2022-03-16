from matplotlib import transforms
from pydantic import Field
import torch
import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.vocabulary import Vocabulary
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
image_transformations = Compose([ToTensor(), Resize((224, 224))])


class BaseDataset(Dataset):
    """
    this is a base dataset to handle both image captioning and concept detection
    """
    def __init__(self, images_dir, transform) -> None:
        self.images_dir = images_dir
        self.transform = transform
        self.available_images = [Path(x).stem for x in glob.glob1(self.images_dir, "*.jpg")]
    
    def pil_image_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class CustomImageDataset(BaseDataset):
    def __init__(self, labels_path, images_dir, vocabulary: Vocabulary, transform=image_transformations, text_transform=None): 
        self.image_labels = pd.read_csv(labels_path, sep="\t")
        super().__init__(self, images_dir, transform)
        self.vocabulary = vocabulary
    
    def __len__(self):
        return self.image_labels.shape[0]

    def __getitem__(self, idx):
        image_name = f"{self.image_labels.iloc[idx, 0]}.jpg"
        image_labels = self.image_labels.iloc[idx, 1]
        image_path = Path(self.images_dir).joinpath(image_name, '')
        image = self.pil_image_loader(str(image_path))
        labels = image_labels.split(";")
        label_vector = self.vocabulary.encode(labels)
        if self.transform:
            image = self.transform(image)
        return image, label_vector

    def pil_image_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class ImageCaptionDataset(BaseDataset):
    def __init__(self,
                 caption_path,
                 images_dir,
                 sequence_length,
                 transform=image_transformations,
                 text_transform=None,
                 text_tokenizer=lambda x: x.split(" ")):
        super().__init__(images_dir, transform)
        self.image_captions = pd.read_csv(caption_path, sep="\t")
        self.image_captions = self.image_captions.loc[self.image_captions.iloc[:, 0].isin(self.available_images)]
        self.sequence_length = sequence_length
        self.text_transform = text_transform
        self.text_tokenizer = text_tokenizer

    def __getitem__(self, idx):
        image_name = f"{self.image_captions.iloc[idx, 0]}.jpg"
        image_caption = self.image_captions.iloc[idx, 1]
        image_path = Path(self.images_dir).joinpath(image_name, '')
        image = self.pil_image_loader(str(image_path))
        processed_captions = self.text_tokenizer(image_caption)
        if self.text_transform:
            processed_captions = self.text_transform(processed_captions)
        if self.transform:
            image = self.transform(image)
        return image, processed_captions
    
    def __len__(self):
        return self.image_captions.shape[0]
