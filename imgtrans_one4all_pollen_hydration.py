import os
import math
import timeit
import argparse
import numpy as np
import utils
import pickle
import random
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import wandb
from torchvision import datasets, models, transforms
import torch.utils.data as data

os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='SCN Pollen')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlpb_hydration')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--dimensions', default=1, type=int)
parser.add_argument('--transform', default='hydration', type=str)
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()

# dataset_root_folder = f'./data/images_7_types_7030'

# dataset_root_folder = 'data/images_3_types_dry_7030'
# dataset_root_folder = 'data/images_3_types_half_hydrated_7030'
# dataset_root_folder = 'data/images_3_types_hydrated_7030'

# train_directory = f'{dataset_root_folder}_train'
# valid_directory = f'{dataset_root_folder}_val'
# test_directory = f'{dataset_root_folder}_test'


dataset_root_folder = './data/images_3_types_hydrated_7030'

train_directory = f'{dataset_root_folder}_train'
train_dry_directory = './data/images_3_types_dry_7030_train'
train_half_hydrated_directory =  './data/images_3_types_half_hydrated_7030_train'


valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'

valid_dry_directory = './data/images_3_types_dry_7030_val'
valid_half_hydrated_directory = './data/images_3_types_half_hydrated_7030_val'


# Applying transforms to the data
image_transforms = {
    'name': 'image_transforms_normal_3',
    'train': transforms.Compose([
        # transforms.Resize(size=img_size + 4),
        # transforms.CenterCrop(size=img_size),
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(
        #     brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                              [0.1434643,  0.16687445, 0.15344492]),
    ]),
    'valid': transforms.Compose([
        # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                             [0.1434643,  0.16687445, 0.15344492]),
    ]),
}


dataset = {
    'train_hydrated': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'train_dry': datasets.ImageFolder(root=train_dry_directory, transform=image_transforms['train']),
    'train_half_hydrated': datasets.ImageFolder(root=train_half_hydrated_directory, transform=image_transforms['train']),
    
    'valid_half_hydrated': datasets.ImageFolder(root=valid_half_hydrated_directory, transform=image_transforms['valid']),    
    'valid_dry': datasets.ImageFolder(root=valid_dry_directory, transform=image_transforms['valid']),    
    
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
}

# Create iterators for data loading
dataloaders = {
    'train_hydrated': data.DataLoader(dataset['train_hydrated'], batch_size=args.batchsize, shuffle=True),
    'train_dry': data.DataLoader(dataset['train_dry'], batch_size=args.batchsize, shuffle=True),
    'train_half_hydrated': data.DataLoader(dataset['train_half_hydrated'], batch_size=args.batchsize, shuffle=True),
    
    'valid_half_hydrated': data.DataLoader(dataset['valid_half_hydrated'], batch_size=args.batchsize, shuffle=False),
    'valid_dry': data.DataLoader(dataset['valid_dry'], batch_size=args.batchsize, shuffle=False),
    
    'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}


def main():
    utils.set_seed(13)
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(dataset['train_hydrated'].classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## download datasets
    # train_loader = dataloaders['train']
    # test_loader = dataloaders['test']
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)  # prints [batch_size, 1, 28, 28] and [batch_size]
    #     break  # remove break to go through all batches

    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    # wandb.init(project=f"SCNPollen", entity="ahinea", name=f"One4All_{args.transform}_{save_dir}")
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))
    
    ######## train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    def reset_iterator(data_loader):
        return iter(data_loader)
    
    def train(model, loss_fn, optimizer):
        # Create iterators for the data loaders
        iterator1 = iter(dataloaders['train_hydrated'])
        iterator2 = iter(dataloaders['train_half_hydrated'])
        iterator3 = iter(dataloaders['train_dry'])
        
        for _ in range(12):  # loop for 10 batches
            # Randomly choose a dataset
            t = random.choice([0, 0.5, 1.0])
            
            if t == 0:
                try:
                    inputs, labels = next(iterator1)
                except StopIteration:
                    iterator1 = reset_iterator(dataloaders['train_hydrated'])
                    inputs, labels = next(iterator1)
            elif t == 0.5:
                try:
                    inputs, labels = next(iterator2)
                except StopIteration:
                    iterator2 = reset_iterator(dataloaders['train_half_hydrated'])
                    inputs, labels = next(iterator2)
            elif t == 1:
                try:
                    inputs, labels = next(iterator3)
                except StopIteration:
                    iterator3 = reset_iterator(dataloaders['train_dry'])
                    inputs, labels = next(iterator3)
            
            X, y = inputs.to(device), labels.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    

    # def validate(dataloader, model, loss_fn):
    #     param = random.uniform(0.2, 2)
    #     model.eval()
    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         iterator1 = iter(dataloaders['test'])
    #         iterator2 = iter(dataloaders['valid_half_hydrated'])
    #         iterator3 = iter(dataloaders['valid_dry'])
            
    #         for _ in range(12):  # loop for 10 batches
    #             # Randomly choose a dataset
    #             t = random.choice([0, 0.5, 1.0])
                
    #             if t == 0:
    #                 try:
    #                     inputs, labels = next(iterator1)
    #                 except StopIteration:
    #                     iterator1 = reset_iterator(dataloaders['test'])
    #                     inputs, labels = next(iterator1)
    #             elif t == 0.5:
    #                 try:
    #                     inputs, labels = next(iterator2)
    #                 except StopIteration:
    #                     iterator2 = reset_iterator(dataloaders['valid_half_hydrated'])
    #                     inputs, labels = next(iterator2)
    #             elif t == 1:
    #                 try:
    #                     inputs, labels = next(iterator3)
    #                 except StopIteration:
    #                     iterator3 = reset_iterator(dataloaders['valid_dry'])
    #                     inputs, labels = next(iterator3)
                
    #             X, y = inputs.to(device), labels.to(device)

    #             pred = model(X)
    #             test_loss += loss_fn(pred, y).item()
    #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    #     test_loss /= len(dataloader)
    #     correct /= len(dataloader.dataset)
    #     print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    #     return correct, test_loss

    for t in range(args.epochs):
        print(f"=================\n Epoch: {t + 1} \n=================")
        train(model, loss_fn, optimizer)
        # test_acc, test_loss = test(model, loss_fn)
        # wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        
    # print("Done!")
    # wandb.finish()

    ######## test model
    def test(model, loss_fn, param):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            iterator1 = iter(dataloaders['test'])
            iterator2 = iter(dataloaders['valid_half_hydrated'])
            iterator3 = iter(dataloaders['valid_dry'])
            
            exhausted_datasets = set()
    
            while len(exhausted_datasets) < 1:
                # Randomly choose a dataset, it's equivalent to image transformation
                t = param
                
                if t == 0 and 0 not in exhausted_datasets:
                    try:
                        inputs, labels = next(iterator1)
                        # print("Batch from Dataset 1")
                    except StopIteration:
                        exhausted_datasets.add(0)
                        continue
                elif t == 0.5 and 0.5 not in exhausted_datasets:
                    try:
                        inputs, labels = next(iterator2)
                        # print("Batch from Dataset 2")
                    except StopIteration:
                        exhausted_datasets.add(0.5)
                        continue
                elif t == 1.0 and 1.0 not in exhausted_datasets:
                    try:
                        inputs, labels = next(iterator3)
                        # print("Batch from Dataset 3")
                    except StopIteration:
                        exhausted_datasets.add(1)
                        continue
                    
                X, y = inputs.to(device), labels.to(device)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        if param == 0:
            test_loss /= len(dataloaders['test'])
            correct /= len(dataloaders['test'].dataset)
        elif param == 0.5:
            test_loss /= len(dataloaders['valid_half_hydrated'])
            correct /= len(dataloaders['valid_half_hydrated'].dataset)
        elif param == 1.0:
            test_loss /= len(dataloaders['valid_dry'])
            correct /= len(dataloaders['valid_dry'].dataset)
                
      
        print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    for param in [0, 0.5, 1.0]:
        acc.append(test(model, loss_fn, param))

    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/One4All/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(acc))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
