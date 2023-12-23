train_df = pd.read_csv('OASIS-1_dataset/tsv_files/lab_1/train.tsv', sep='\t')
valid_df = pd.read_csv('OASIS-1_dataset/tsv_files/lab_1/validation.tsv', sep='\t')
train_population_df = characteristics_table(train_df, OASIS_df)
valid_population_df = characteristics_table(valid_df, OASIS_df)
print(f"Train dataset:\n {train_population_df}\n")
print(f"Validation dataset:\n {valid_population_df}")


img_dir = path.join('OASIS-1_dataset', 'CAPS')
batch_size=4

example_dataset = MRIDataset(img_dir, OASIS_df, transform=CropLeftHC())
example_dataloader = DataLoader(example_dataset, batch_size=batch_size, drop_last=True)
for data in example_dataloader:
    pass

print(f"Shape of Dataset output:\n {example_dataset[0]['image'].shape}\n")

print(f"Shape of DataLoader output:\n {data['image'].shape}")


from torch import nn

conv_layer = nn.Conv3d(8, 16, 3)
print('Weights shape\n', conv_layer.weight.shape)
print()
print('Bias shape\n', conv_layer.bias.shape)

batch_layer = nn.BatchNorm3d(16)
print('Gamma value\n', batch_layer.state_dict()['weight'].shape)
print()
print('Beta value\n', batch_layer.state_dict()['bias'].shape)
