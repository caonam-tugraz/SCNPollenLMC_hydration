import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
from collections import defaultdict
import numpy as np

# Define the paths to the dry and hydrated datasets
dry_folder = 'data/images_3_types_dry_7030_train'
hydrated_folder = 'data/images_3_types_hydrated_7030_train'

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Resize the images to a fixed size
    transforms.ToTensor()         # Convert images to tensor
])

# Create the datasets
dry_dataset = datasets.ImageFolder(root=dry_folder, transform=transform)
hydrated_dataset = datasets.ImageFolder(root=hydrated_folder, transform=transform)

# Create data loaders for the datasets (adjust the batch size as needed)
batch_size = 8
dry_loader = DataLoader(dry_dataset, batch_size=batch_size, shuffle=True)
hydrated_loader = DataLoader(hydrated_dataset, batch_size=1, shuffle=True)

# Load one batch from the dry dataset
dry_batch = next(iter(dry_loader))
dry_images, dry_labels = dry_batch

print("Dry batch labels:", dry_labels.tolist())

# Create a dictionary to store the indices of hydrated dataset by class
hydrated_class_indices = defaultdict(list)
for idx, (_, label) in enumerate(hydrated_dataset):
    hydrated_class_indices[label].append(idx)

# Function to sample a batch from hydrated set based on dry batch labels
def sample_hydrated_batch(dry_labels, hydrated_class_indices, hydrated_dataset):
    hydrated_batch_indices = []
    for label in dry_labels:
        if hydrated_class_indices[int(label)]:
            hydrated_index = np.random.choice(hydrated_class_indices[int(label)], 1, replace=False)[0]
            hydrated_batch_indices.append(hydrated_index)
    
    # Create a subset of the hydrated dataset based on the sampled indices
    hydrated_batch = torch.utils.data.Subset(hydrated_dataset, hydrated_batch_indices)
    hydrated_loader = DataLoader(hydrated_batch, batch_size=len(hydrated_batch_indices), shuffle=False)
    
    # Load the hydrated batch
    hydrated_images, hydrated_labels = next(iter(hydrated_loader))
    
    return hydrated_images, hydrated_labels

# Sample the hydrated batch
hydrated_images, hydrated_labels = sample_hydrated_batch(dry_labels, hydrated_class_indices, hydrated_dataset)

print("Hydrated batch labels:", hydrated_labels.tolist())
