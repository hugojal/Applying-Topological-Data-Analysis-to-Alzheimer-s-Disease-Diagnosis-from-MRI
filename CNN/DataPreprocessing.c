from torch.utils.data import Dataset, DataLoader, sampler
from os import path

class MRIDataset(Dataset):

    def __init__(self, img_dir, data_df, transform=None):
        """
        Args:
            img_dir (str): path to the CAPS directory containing preprocessed images
            data_df (DataFrame): metadata of the population.
                Columns include participant_id, session_id and diagnosis).
            transform (callable): list of transforms applied on-the-fly, chained with torchvision.transforms.Compose.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data_df = data_df
        self.label_code = {"AD": 1, "CN": 0}

        self.size = self[0]['image'].shape

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):

        diagnosis = self.data_df.loc[idx, 'diagnosis']
        label = self.label_code[diagnosis]

        participant_id = self.data_df.loc[idx, 'participant_id']
        session_id = self.data_df.loc[idx, 'session_id']
        filename = 'subjects/' + participant_id + '/' + session_id + '/' + \
          'deeplearning_prepare_data/image_based/custom/' + \
          participant_id + '_' + session_id + \
          '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt'

        image = torch.load(path.join(self.img_dir, filename))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant_id,
                  'session_id': session_id}
        return sample

    def train(self):
        self.transform.train()

    def eval(self):
        self.transform.eval()


class CropLeftHC(object):
    """Crops the left hippocampus of a MRI non-linearly registered to MNI"""
    def __init__(self, random_shift=0):
        self.random_shift = random_shift
        self.train_mode = True
    def __call__(self, img):
        if self.train_mode:
            x = random.randint(-self.random_shift, self.random_shift)
            y = random.randint(-self.random_shift, self.random_shift)
            z = random.randint(-self.random_shift, self.random_shift)
        else:
            x, y, z = 0, 0, 0
        return img[:, 25 + x:55 + x,
                   50 + y:90 + y,
                   27 + z:57 + z].clone()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


class CropRightHC(object):
    """Crops the right hippocampus of a MRI non-linearly registered to MNI"""
    def __init__(self, random_shift=0):
        self.random_shift = random_shift
        self.train_mode = True
    def __call__(self, img):
        if self.train_mode:
            x = random.randint(-self.random_shift, self.random_shift)
            y = random.randint(-self.random_shift, self.random_shift)
            z = random.randint(-self.random_shift, self.random_shift)
        else:
            x, y, z = 0, 0, 0
        return img[:, 65 + x:95 + x,
                   50 + y:90 + y,
                   27 + z:57 + z].clone()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate

subject = 'sub-OASIS10003'
preprocessed_pt = torch.load(f'OASIS-1_dataset/CAPS/subjects/{subject}/ses-M00/' +
                    f'deeplearning_prepare_data/image_based/custom/{subject}_ses-M00_' +
                    'T1w_segm-graymatter_space-Ixi549Space_modulated-off_' +
                    'probability.pt')
raw_nii = nib.load(f'OASIS-1_dataset/raw/{subject}_ses-M00_T1w.nii.gz')

raw_np = raw_nii.get_fdata()

def show_slices(slices):
    """ Function to display a row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = raw_np[80, :, :]
slice_1 = raw_np[:, 130, :]
slice_2 = raw_np[:, :, 60]
show_slices([rotate(slice_0, 90), rotate(slice_1, 90), slice_2])
plt.suptitle(f'Slice of raw image of subject {subject}')
plt.show()

slice_0 = preprocessed_pt[0, 60, :, :]
slice_1 = preprocessed_pt[0, :, 72, :]
slice_2 = preprocessed_pt[0, :, :, 60]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of preprocessed image of subject {subject}')
plt.show()

leftHC_pt = CropLeftHC()(preprocessed_pt)
slice_0 = leftHC_pt[0, 15, :, :]
slice_1 = leftHC_pt[0, :, 20, :]
slice_2 = leftHC_pt[0, :, :, 15]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of left HC of subject {subject}')
plt.show()

