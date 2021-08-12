from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import pickle


class SegmentationDataset(Dataset):
    pass


class ObjectDetectionDataset(Dataset):
    pass


class ClassificationDataset(Dataset):
    def __init__(self,
                 p_data,
                 transform: callable = None,
                 target_transform: callable = None,
                 split='train'):

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.p_data = Path(p_data)
        self.p_config = self.p_data / "config.pkl"
        assert self.p_config.exists()

        with open(self.p_config, "rb") as file:
            self.config = pickle.load(file)

        self.p_img_folder = self.p_data / self.config["p_imgs"]
        assert self.p_img_folder.exists()

        if self.split == "train":
            self.p_labels = self.p_data / self.config["p_Y_train_clas"]
            self.p_imgs = self.p_data / self.config["p_X_train"]
        else:
            self.p_labels = self.p_data / self.config["p_Y_valid_clas"]
            self.p_imgs = self.p_data / self.config["p_X_valid"]

        assert self.p_imgs.exists()
        assert self.p_labels.exists()
        assert self.p_img_folder.exists()

        self.X = np.load(self.p_imgs)
        self.Y = np.load(self.p_labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = Image.open(self.p_img_folder / self.X[idx])
        y = self.Y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
