import torchvision
from torchvision import transforms as T
import torch

class DataLoader(object):

    def __init__(self, args):
        """
        This class is preparing data for feeding into torch-model
        """
        transforms = [T.ToTensor()]
        if args.random_flip_prob:
            transforms.append(T.RandomHorizontalFlip(p=args.random_flip_prob))

        self.transforms = T.Compose(transforms)

        self.tr_root = args.tr_root
        self.val_root = args.val_root
        self.test_root = args.test_root

        self.batch_size = args.batch_size

    def setup(self):
        self.tr_data = torchvision.datasets.ImageFolder(root=self.tr_root, transform=self.transforms)
        self.val_data = torchvision.datasets.ImageFolder(root=self.val_root, transform=self.transforms)
        self.test_data = torchvision.datasets.ImageFolder(root=self.test_root, transform=self.transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.tr_data, batch_size=self.batch_size,
                                          shuffle=True, num_workers=4,
                                          pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)
