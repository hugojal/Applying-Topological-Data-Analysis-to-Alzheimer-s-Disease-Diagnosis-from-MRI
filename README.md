class CropMaxUnpool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool3d, self).__init__()
        self.unpool = nn.MaxUnpool3d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[4]
            y1 = padding[2]
            z1 = padding[0]
            output = output[:, :, x1::, y1::, z1::]

        return output


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Initial size (30, 40, 30)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, return_indices=True, return_pad=True),
            # Size (15, 20, 15)

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, return_indices=True, return_pad=True),
            # Size (8, 10, 8)

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, return_indices=True, return_pad=True),
            # Size (4, 5, 4)

            nn.Conv3d(32, 1, 1),
            # Size (4, 5, 4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1, 32, 1),
            # Size (4, 5, 4)

            CropMaxUnpool3d(2, 2),
            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            # Size (8, 10, 8)

            CropMaxUnpool3d(2, 2),
            nn.ConvTranspose3d(16, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            # Size (15, 20, 15)

            CropMaxUnpool3d(2, 2),
            nn.ConvTranspose3d(8, 1, 3, padding=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
            # Size (30, 40, 30)
        )

    def forward(self, x):
        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if isinstance(layer, PadMaxPool3d):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            else:
                x = layer(x)

        code = x.view(x.size(0), -1)
        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            else:
                x = layer(x)

        return code, x



# To complete
def trainAE(model, train_loader, criterion, optimizer, n_epochs):
    """
    Method used to train an AutoEncoder

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network.
    """
    best_model = deepcopy(model)
    train_best_loss = np.inf

    for epoch in range(n_epochs):
        model.train()
        train_loader.dataset.train()
    for i, data in enumerate(train_loader, 0):
        mean_loss = testAE(model, train_loader, criterion)

        print(f'Epoch %i: loss = %f' % (epoch, mean_loss))

        if mean_loss < train_best_loss:
            best_model = deepcopy(model)
            train_best_loss = mean_loss

    return best_model


def testAE(model, data_loader, criterion):
    """
    Method used to test an AutoEncoder

    Args:
        model: (nn.Module) the neural network
        data_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()
    data_loader.dataset.eval()
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data['image'].cuda(), data['label'].cuda()
            _, outputs = model((images))
            loss = criterion(outputs, images)
            total_loss += loss.item()

    return total_loss / len(data_loader.dataset) / np.product(data_loader.dataset.size)


    !pip install autoencoders
!pip install torchvision


learning_rate = 10**-2
n_epochs = 30
batch_size = 4

class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, padding, return_indices=False, return_pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_indices = return_indices
        self.return_pad = return_pad

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = F.max_pool3d(x, self.kernel_size, self.stride, return_indices=self.return_indices)
        if self.return_pad:
            return x, self.padding
        else:
            return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, padding=True, return_indices=True, return_pad=True),
            # Size (15, 20, 15)
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, padding=True, return_indices=True, return_pad=True),
            # Size (7, 10, 7)
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2, padding=True, return_indices=True, return_pad=True),
            # Size (3, 5, 3)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
        )

AELeftHC = AutoEncoder().cuda()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(AELeftHC.parameters(), learning_rate)




import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate

subject = 'sub-OASIS10003'
preprocessed_pt = torch.load(f'OASIS-1_dataset/CAPS/subjects/{subject}/ses-M00/' +
                    'deeplearning_prepare_data/image_based/custom/' + subject +
                    '_ses-M00_'+
                    'T1w_segm-graymatter_space-Ixi549Space_modulated-off_' +
                    'probability.pt')
input_pt = CropLeftHC()(preprocessed_pt).unsqueeze(0).cuda()
_, output_pt = best_AELeftHC(input_pt)


slice_0 = input_pt[0, 0, 15, :, :].cpu()
slice_1 = input_pt[0, 0, :, 20, :].cpu()
slice_2 = input_pt[0, 0, :, :, 15].cpu()
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of the input image of subject {subject}')
plt.show()

slice_0 = output_pt[0, 0, 15, :, :].cpu().detach()
slice_1 = output_pt[0, 0, :, 20, :].cpu().detach()
slice_2 = output_pt[0, 0, :, :, 15].cpu().detach()
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of the output image of subject {subject}')
plt.show()
