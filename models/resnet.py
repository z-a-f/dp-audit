r'''PyTorch Lightning modules for the ResNet models'''

import torch
import torchvision

import lightning.pytorch as pl

class ResNetModule(pl.LightningModule):
    r'''PyTorch Lightning module for the ResNet model'''
    def __init__(self, variant, num_classes):
        super().__init__()
        if isinstance(variant, int):
            variant = f'resnet{variant}'
        variant = variant.lower()
        self.model = getattr(torchvision.models, f'{variant}')(weights='DEFAULT')
        if num_classes is None:
            raise ValueError('num_classes must be specified. '
                             'Please use "--model.num_classes" in the CLI or "num_classes" in the config yaml')
        if self.model is None:
            raise ValueError(f'Unsupported ResNet variant: {variant}')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        # self.learning_rate = learning_rate

    def forward(self, x):
        r'''Forward pass'''
        return self.model(x)

    def training_step(self, batch, batch_idx):
        r'''Training step'''
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        r'''Validation step'''
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        r'''Configure the optimizer
        
        This is the default optimizer configuration for the ResNet model.
        Ideally, you would want to specify an optimizer in the configuration file.
        '''
        learning_rate = 0.1
        momentum=0.9
        weight_decay=5e-4

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
        return [optimizer], [scheduler]