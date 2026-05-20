
import dataset

# create training and validation datasets
train_dataset = dataset.CSE144Dataset(dataset.train_images, dataset.train_labels, transform=dataset.train_transform)
val_dataset = dataset.CSE144Dataset(dataset.val_images, dataset.val_labels, transform=dataset.val_transform)

# run unit tests
dataset.dataset_unittest(train_dataset)
dataset.dataset_unittest(val_dataset)

# create DataLoaders
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_dataloader = dataset.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = dataset.DataLoader(val_dataset, batch_size=32, shuffle=True)
