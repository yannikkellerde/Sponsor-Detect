from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os
import torch

CATEGORY_MAP = {
    "video": 0,
    "sponsor": 1,
}

class video_image_dataset(Dataset):
    
    def __init__(self, dataset_df, path) -> None:
        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        self.path = path
        self.files = dataset_df["frame_name"].tolist()
        label = dataset_df["label"].tolist()
        self.labels = [CATEGORY_MAP[l] for l in label]

    def load(self, file):
        return self.preprocess(Image.open(os.path.join(self.path, file)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return (self.load(self.files[index]), self.labels[index])
    
