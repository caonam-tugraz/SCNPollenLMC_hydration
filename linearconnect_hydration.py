import argparse
import os
import pickle
import random
import timeit

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

import utils
# import wandb

os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='Qcurves')
parser.add_argument('--dataset', default='FashionMNIST', type=str,
                    help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--arch', '-a', default='sconvb_hydration', type=str) # sconvb_hydration, mlpb_hydration
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--transform', default='hydration', type=str)
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()


# dataset_root_folder = './data/images_7_types_7030'

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
                             [0.1434643, 0.16687445, 0.15344492]),
    ]),
    'valid': transforms.Compose([
        # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                             [0.1434643, 0.16687445, 0.15344492]),
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
    
    # Create iterators for the data loaders
    iterator1 = iter(dataloaders['train_hydrated'])
    iterator2 = iter(dataloaders['train_half_hydrated'])
    iterator3 = iter(dataloaders['train_dry'])


    utils.set_seed(13)
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(dataset['train_hydrated'].classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## download datasets
    # train_loader = dataloaders['train_half_hydrated']
    # test_loader = dataloaders['test']
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)  # prints [batch_size, 1, 28, 28] and [batch_size]
    #     break  # remove break to go through all batches

    ######## prepare model structure
    # params = [0.2, 2.0]
    m0, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m0.to(device)
    m0.train()
    print(m0)
    print(utils.count_model_parameters(m0))

    m1, _ = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m1.to(device)
    m1.train()

    ######## train model
    
    
    lfn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam([*m0.parameters(), *m1.parameters()], lr=args.learning_rate)
    
    def reset_iterator(data_loader):
        return iter(data_loader)

    for ep in range(args.epochs):
        # Initialize iterators
        iterator1 = reset_iterator(dataloaders['train_hydrated'])
        iterator2 = reset_iterator(dataloaders['train_half_hydrated'])
        iterator3 = reset_iterator(dataloaders['train_dry'])
        
        for _ in range(12):  # loop for 10 batches

            # Randomly choose a dataset
            loss = 0
            for _ in range(10):  # Accumulate gradients over 10 batches
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
                
                inputs, labels = inputs.to(device), labels.to(device)
    
                logits = m0.linearcurve(m0, m1, t, inputs)
                loss += lfn(logits, labels)
    
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            print(f"epoch {ep} step loss: {loss}")
    
        print(f"epoch {ep} completed")
    
        # wandb.log({"train/loss": loss})

    destination_name = f'{args.output}/{args.transform}/LinearConnect/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)

    torch.save(m0.state_dict(), f'{destination_name}/linearconnect_m0.pt')
    torch.save(m1.state_dict(), f'{destination_name}/linearconnect_m1.pt')

    print('Time: ', timeit.default_timer() - start)
    # wandb.finish()


    #######################################
    ############# Evaluate
    #######################################
    # evaluates acc and loss
    def evaluate_param(model, loader):
        model = model.cuda()
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs.cuda())
                pred = outputs.argmax(dim=1)
                correct += (labels.cuda() == pred).sum().item()
                total += len(labels)
                loss = F.cross_entropy(outputs, labels.cuda())
                losses.append(loss.item())
        return correct / total, np.array(losses).mean()

    print('LinearConnect:')

    linearconnect = []
    # for t in trange:
        
        
    train_dataset_list = ['train_hydrated', 'train_half_hydrated', 'train_dry']
    eval_dataset_list = ['test', 'valid_half_hydrated', 'valid_dry']
    
    for t, dts_name, evdts_name in zip([0, 0.5, 1.0], train_dataset_list, eval_dataset_list) :
        # param = ((1 - t) * params[0] + t * params[1]) % 360
        mtmp, _ = utils.prepare_model(args, nchannels, nclasses)
        mtmp_sd = {k: (((1-t) * m0.state_dict()[k].cpu() +
                          t * m1.state_dict()[k].cpu())
        ) for k in m0.state_dict().keys()}
        mtmp.load_state_dict(mtmp_sd)
    
        # run a single train epoch with augmentations to recalc stats
        mtmp.train()
        with torch.no_grad():
            
            for images, _ in dataloaders[dts_name]:
                mtmp(images)
        mtmp.eval()
    
    
        acc, loss = evaluate_param(mtmp.cuda(), loader=dataloaders[evdts_name])
        
        # Eval on the other remaining "transformations" 
        remaining_dts = [elem for elem in eval_dataset_list if elem != evdts_name]
        
        acc1, loss1 = evaluate_param(mtmp.cuda(), loader=dataloaders[remaining_dts[0]])
        print(remaining_dts[0])
        acc2, loss2 = evaluate_param(mtmp.cuda(), loader=dataloaders[remaining_dts[1]])
        print(remaining_dts[1])
        
        print(f"{t} --> {[acc, acc1, acc2]}")
        # print(f"{t} --> {[acc]}")
        linearconnect.append([1, acc, acc1, acc2])

    stats = {'linearconnect': linearconnect}
    np.save(f'{destination_name}/acc.npy', pickle.dumps(stats))


if __name__ == '__main__':
    main()
