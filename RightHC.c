class CropRightHC(object):
    """
    Crops the right half of the image.

    Args:
        crop_size (int): The size of the cropped image.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        """
        Crops the right half of the image.

        Args:
            image (torch.Tensor): The image to crop.

        Returns:
            torch.Tensor: The cropped image.
        """

        return image[:, :, :self.crop_size]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def train(model, train_loader, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Epoch: {} Batch: {} Loss: {}'.format(epoch, batch_idx, loss.item()))

    return model

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to('cuda'), target.to('cuda')
        output = model(data)
        loss


# To complete
def softvoting(leftHC_df, rightHC_df):
    valid_results = softvoting(valid_resultsLeftHC_df, valid_resultsRightHC_df)
    valid_metrics = compute_metrics(valid_results.true_label, valid_results.predicted_label)
