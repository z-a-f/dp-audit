r'''Perform a membership inference attack on a model.'''

import abc

import numpy as np
from sklearn import linear_model, model_selection

import torch

from models.resnet import ResNetModule
from data.cifar import CIFARDataModule

class _BaseObserver(abc.ABC):
    r'''Base class for the observer.
    
    An observer is an object that analyzes the model, given current inputs.
    The simplest observer is just the model itself, which returns the model's output.
    A slightly more complex version could be returning per-sample loss values, given the inputs.
    '''
    def __init__(self, model, *args, **kwargs):
        self.model = model
        # Infer the device from the model's parameters
        self.device = next(model.parameters()).device
    
    @abc.abstractmethod
    def observe(self, dataloader, *args, **kwargs):
        r'''Observe the model's behavior.
        
        Returns:
            The observed value.
        '''
        pass

    def __call__(self, *args, **kwargs):
        return self.observe(*args, **kwargs)

class PerSampleLossObserver(_BaseObserver):
    def __init__(self, model, criterion, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.criterion = criterion
        if hasattr(self.criterion, 'reduction'):
            self.criterion.reduction = 'none'
    
    @torch.no_grad()
    def observe(self, dataloader, *args, **kwargs):
        is_training = self.model.training
        self.model.train(False)
        losses = []
        for batch in dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)  # Shape: (batch_size,)
            losses.append(loss)
        self.model.train(is_training)
        return torch.hstack(losses)  # Shape: (num_samples,)


def simple_mia(observer, seen_dataloader, unseen_dataloader, n_splits=5, random_state=0):
    '''Computes cross-validation score of a membership inference attack.

    Based on https://github.com/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb

    Args:
        observer: The observer object.
        seen_dataloader: The dataloader for the data that was seen during training.
        unseen_dataloader: The dataloader for the that was not seen during training (s.a. validation or test set).
        n_splits: The number of splits for the cross-validation.
        random_state: The random state for the cross-validation.
    '''
    # Create training dataset
    member_observations = observer(seen_dataloader)
    nonmember_observations = observer(unseen_dataloader)
    member_observations = member_observations[:len(nonmember_observations)]
    # print(f'===> DEBUG: member stats: {member_observations.mean()}, {member_observations.std()}')
    # print(f'===> DEBUG: nonmember stats: {nonmember_observations.mean()}, {nonmember_observations.std()}')
    num_features = member_observations.shape[1] if member_observations.ndim > 1 else 1
    sample_loss = torch.cat([member_observations, nonmember_observations]).cpu().numpy().reshape(-1, num_features)
    members = np.concatenate([np.ones(len(member_observations)), np.zeros(len(nonmember_observations))])

    # Check that the observer returns the right shape
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    # Train the attack model
    attack_model = linear_model.LogisticRegression(class_weight="balanced")
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )

    # Compute the cross-validation score for the MIA
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


class MIA:
    def __init__(self):
        pass

    def mia(self, observer, model, data, n_splits=5, random_state=0):
        '''Computes cross-validation score of a membership inference attack.'''
        


# def main():
#     model_path = 'lightning_logs/70lo6r2i/checkpoints/epoch=171-val_loss=0.4661.ckpt'
#     model = ResNetModule.load_from_checkpoint(model_path).to('cuda')
#     data = CIFARDataModule('CIFAR10').prepare_data().setup('fit')
#     # Hack to remove the random transformations from the training set
#     data.train_dataset.transform = data.val_dataset.transform

#     observer = PerSampleLossObserver(model, torch.nn.functional.cross_entropy)
#     train_dataloader = data.train_dataloader()
#     val_dataloader = data.val_dataloader()
    
#     # Perform the membership inference attack
#     mia_score = simple_mia(observer, train_dataloader, val_dataloader)
#     print(f'Membership inference attack score: {mia_score.mean():.2%}')
        
def main():
    import jsonargparse
    import lightning.pytorch
    parser = jsonargparse.ArgumentParser(fail_untyped=False)

    parser.add_argument('--model', type=lightning.pytorch.LightningModule, required=True)
    parser.add_argument('--data', type=lightning.pytorch.LightningDataModule, required=True)
    parser.add_argument('--observer', type=_BaseObserver, required=True)

    args = parser.parse_args()
    print(args)

if __name__ == '__main__':
    main()