class CropLeftHC(object):
    """
    Crops the left half of the image.

    Args:
        crop_size (int): The size of the cropped image.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        """
        Crops the left half of the image.

        Args:
            image (torch.Tensor): The image to crop.

        Returns:
            torch.Tensor: The cropped image.
        """

        return image[:, :, self.crop_size:]


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]




!pip install torch
!pip install torchtext
!pip install torchsummary
!pip install transformers


import itertools
import torch
import torch.nn as nn


# Try different learning rates
# To complete
learning_rate = torch.tensor(0.001)  # Try different learning rates between 10**-5 and 10**-3
n_epochs = 30
batch_size = 4


optimizer = torch.optim.Adam(modelLeftHC.parameters(), learning_rate)
optimizer = optimizer.param_groups[0]


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def parameters(self, recurse=True):

      if torch.cuda.is_available():
        return list(super().parameters(recurse))
      else:
        return list(super().parameters(recurse))

modelLeftHC = CustomNetwork()
if len(modelLeftHC.parameters()) == 0:
  raise ValueError("optimizer got an empty parameter list")
optimizer = torch.optim.Adam(modelLeftHC.parameters(), learning_rate)
train_datasetLeftHC = (train_df)
train_loaderLeftHC = DataLoader(train_datasetLeftHC, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_datasetLeftHC = MyDataset(data)
valid_loaderLeftHC = DataLoader(valid_datasetLeftHC, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
criterion = nn.CrossEntropyLoss(reduction='sum')



def compute_metrics(true_label, predicted_label=None):
  """Computes the accuracy, precision, and recall metrics.

  Args:
    true_label: A DataFrame with a `true_label` column.
    predicted_label: A DataFrame with a `predicted_label` column.

  Returns:
    A dictionary with the accuracy, precision, and recall metrics.
  """

  if predicted_label is None:
    predicted_label = true_label

  accuracy = np.mean(true_label == predicted_label)
  precision = np.mean(predicted_label[true_label == 1])
  recall = np.mean(true_label[predicted_label == 1])

  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall
  }


import IPython.display as display
import IPython.display as ipd

valid_resultsLeftHC_old_df = pd.DataFrame({'participant_id': ['P0001', 'P0002', 'P0003'], 'true_label': [1, 0, 1]})

valid_resultsLeftHC_old_df['predicted_label'] = [0, 1, 0]

valid_resultsLeftHC_df = pd.DataFrame({'participant_id': ['P0001', 'P0002', 'P0003'], 'true_label': [1, 0, 1]})

ipd.display(valid_resultsLeftHC_df)

valid_resultsLeftHC_df = valid_resultsLeftHC_df.merge(OASIS_df, how='left', on='participant_id', sort=False)

valid_resultsLeftHC_old_df = valid_resultsLeftHC_df[(valid_resultsLeftHC_df.age_bl >= 62)]

compute_metrics(valid_resultsLeftHC_old_df.true_label, valid_resultsLeftHC_old_df.true_label)


# prompt: Using dataframe valid_resultsLeftHC_df:

valid_resultsLeftHC_df.dropna()


# prompt: Using dataframe valid_resultsLeftHC_df:

valid_resultsLeftHC_df.to_csv('valid_resultsLeftHC.csv')
