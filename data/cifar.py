r'''PyTorch Lightning data modules for the CIFAR datasets'''

import torch
import torchvision

import lightning.pytorch as pl

class CIFARDataModule(pl.LightningDataModule):
    r'''PyTorch Lightning data module for the CIFAR-X dataset.
    
    Depending on the 'num_classes' argument, either CIFAR-10 or CIFAR-100 will be loaded.
    
    '''
    def __init__(self, variant, data_dir: str = '.data/', batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if isinstance(variant, int):
            variant = f'CIFAR{variant}'
        variant = variant.upper()
        self.dataset_cls = getattr(torchvision.datasets, f'{variant}')
        if self.dataset_cls is None:
            raise ValueError(f'Unsupported CIFAR variant: {variant}')

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def prepare_data(self):
        r'''Downloads the CIFAR-10 dataset'''
        train_set = self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)
        self.mean = train_set.data.mean(axis=(0, 1, 2)) / 255
        self.std = train_set.data.std(axis=(0, 1, 2)) / 255

    def setup(self, stage=None):
        r'''Loads the CIFAR-10 dataset'''
        infer_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])

        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_cls(self.data_dir, train=True, download=False, transform=train_transform)
            self.val_dataset = self.dataset_cls(self.data_dir, train=False, download=False, transform=infer_transform)

        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_cls(self.data_dir, train=False, download=False, transform=infer_transform)

    def train_dataloader(self):
        r'''Return the training dataloader'''
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        r'''Return the validation dataloader'''
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        r'''Return the test dataloader'''
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)