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
from datetime import datetime

import utils

os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='Qcurves')
parser.add_argument('--dataset', default='FashionMNIST', type=str,
                    help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--arch', '-a', default='sconvb_ensemble', type=str) # sconvb_ensemble, mlpb_ensemble
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--transform', default='ensemble', type=str)
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()

# Brightness dataset
brightness_dataset_root_folder = './data/images_3_types_hydrated_7030'
brightness_train_directory = f'{brightness_dataset_root_folder}_train'
brightness_valid_directory = f'{brightness_dataset_root_folder}_val'
brightness_test_directory = f'{brightness_dataset_root_folder}_test'

# Hydration dataset
hydration_dataset_root_folder = './data/images_3_types_hydrated_7030'
hydration_train_directory = f'{hydration_dataset_root_folder}_train'
hydration_train_dry_directory = './data/images_3_types_dry_7030_train'
hydration_train_half_hydrated_directory =  './data/images_3_types_half_hydrated_7030_train'
hydration_valid_directory = f'{hydration_dataset_root_folder}_val'
hydration_test_directory = f'{hydration_dataset_root_folder}_test'
hydration_valid_dry_directory = './data/images_3_types_dry_7030_val'
hydration_valid_half_hydrated_directory = './data/images_3_types_half_hydrated_7030_val'

# Applying transforms to the data
image_transforms = {
    'name': 'image_transforms_normal_3',
    'train': transforms.Compose([
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                             [0.1434643, 0.16687445, 0.15344492]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(100),
        transforms.RandomCrop(96, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                             [0.1434643, 0.16687445, 0.15344492]),
    ]),
}

brightness_dataset = {
    'train': datasets.ImageFolder(root=brightness_train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=brightness_valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=brightness_valid_directory, transform=image_transforms['valid']),
}

hydration_dataset = {
    'train_hydrated': datasets.ImageFolder(root=hydration_train_directory, transform=image_transforms['train']),
    'train_dry': datasets.ImageFolder(root=hydration_train_dry_directory, transform=image_transforms['train']),
    'train_half_hydrated': datasets.ImageFolder(root=hydration_train_half_hydrated_directory, transform=image_transforms['train']),
    'valid_half_hydrated': datasets.ImageFolder(root=hydration_valid_half_hydrated_directory, transform=image_transforms['valid']),
    'valid_dry': datasets.ImageFolder(root=hydration_valid_dry_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=hydration_valid_directory, transform=image_transforms['valid']),
}

# Create iterators for data loading
brightness_dataloaders = {
    'train': data.DataLoader(brightness_dataset['train'], batch_size=args.batchsize, shuffle=True),
    'valid': data.DataLoader(brightness_dataset['valid'], batch_size=args.batchsize, shuffle=False),
    'test': data.DataLoader(brightness_dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}

hydration_dataloaders = {
    'train_hydrated': data.DataLoader(hydration_dataset['train_hydrated'], batch_size=args.batchsize, shuffle=True),
    'train_dry': data.DataLoader(hydration_dataset['train_dry'], batch_size=args.batchsize, shuffle=True),
    'train_half_hydrated': data.DataLoader(hydration_dataset['train_half_hydrated'], batch_size=args.batchsize, shuffle=True),
    'valid_half_hydrated': data.DataLoader(hydration_dataset['valid_half_hydrated'], batch_size=args.batchsize, shuffle=False),
    'valid_dry': data.DataLoader(hydration_dataset['valid_dry'], batch_size=args.batchsize, shuffle=False),
    'test': data.DataLoader(hydration_dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}

def main():
    # Create iterators for the hydration data loaders
    iterator1 = iter(hydration_dataloaders['train_hydrated'])
    iterator2 = iter(hydration_dataloaders['train_half_hydrated'])
    iterator3 = iter(hydration_dataloaders['train_dry'])

    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(hydration_dataset['train_hydrated'].classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## prepare model structure
    m_brightness0, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m_brightness0.to(device)
    m_brightness0.train()
    print(m_brightness0)
    print(utils.count_model_parameters(m_brightness0))

    m_brightness1, _ = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m_brightness1.to(device)
    m_brightness1.train()
    print(m_brightness1)
    print(utils.count_model_parameters(m_brightness1))
    
    m_hydration0, _ = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m_hydration0.to(device)
    m_hydration0.train()

    m_hydration1, _ = utils.prepare_model(args, nchannels, nclasses, hin=1)
    m_hydration1.to(device)
    m_hydration1.train()
    
    ######## train models
    lfn = torch.nn.CrossEntropyLoss()
    # opt_brightness = torch.optim.Adam(m_brightness.parameters(), lr=args.learning_rate)
    opt_brightness = torch.optim.Adam([*m_brightness0.parameters(), *m_brightness1.parameters()], lr=args.learning_rate)
    
    opt_hydration = torch.optim.Adam([*m_hydration0.parameters(), *m_hydration1.parameters()], lr=args.learning_rate)
    params = [0.2, 2.0]

    def reset_iterator(data_loader):
        return iter(data_loader)

    for ep in range(args.epochs):
        # Initialize iterators
        iterator1 = reset_iterator(hydration_dataloaders['train_hydrated'])
        iterator2 = reset_iterator(hydration_dataloaders['train_half_hydrated'])
        iterator3 = reset_iterator(hydration_dataloaders['train_dry'])

        for inputs, labels in brightness_dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            loss_brightness = 0
            for _ in range(10):
                
                t = random.random()
                param = ((1 - t) * params[0] + t * params[1]) % 360
                # Train brightness model
                opt_brightness.zero_grad()
                inputs_transformed = TF.adjust_brightness(inputs, brightness_factor=param)
                logits = m_brightness0.linearcurve(m_brightness0, m_brightness1, t, inputs_transformed)
                loss_brightness += lfn(logits, labels)
                
            loss_brightness.backward()
            opt_brightness.step()

        for _ in range(12):  # loop over all the classes
            # Train hydration model
            
            loss_hydration = 0
            for _ in range(10):  # Accumulate gradients over 10 batches
                t = random.choice([0, 0.5, 1.0])
                if t == 0:
                    try:
                        inputs, labels = next(iterator1)
                    except StopIteration:
                        iterator1 = reset_iterator(hydration_dataloaders['train_hydrated'])
                        inputs, labels = next(iterator1)
                elif t == 0.5:
                    try:
                        inputs, labels = next(iterator2)
                    except StopIteration:
                        iterator2 = reset_iterator(hydration_dataloaders['train_half_hydrated'])
                        inputs, labels = next(iterator2)
                elif t == 1:
                    try:
                        inputs, labels = next(iterator3)
                    except StopIteration:
                        iterator3 = reset_iterator(hydration_dataloaders['train_dry'])
                        inputs, labels = next(iterator3)

                inputs, labels = inputs.to(device), labels.to(device)
                logits = m_hydration0.linearcurve(m_hydration0, m_hydration1, t, inputs)
                loss_hydration += lfn(logits, labels)

            opt_hydration.zero_grad()
            loss_hydration.backward()
            opt_hydration.step()

            print(f"epoch {ep} step loss_brightness: {loss_brightness}, step loss_hydration: {loss_hydration}")

        print(f"epoch {ep} completed")

    destination_name = f'{args.output}/ensemble/LinearConnect/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)

    torch.save(m_brightness0.state_dict(), f'{destination_name}/ensemble_brightness0.pt')
    torch.save(m_hydration0.state_dict(), f'{destination_name}/ensemble_hydration0.pt')
    torch.save(m_brightness1.state_dict(), f'{destination_name}/ensemble_brightness1.pt')
    torch.save(m_hydration1.state_dict(), f'{destination_name}/ensemble_hydration1.pt')
    

    print('Time: ', timeit.default_timer() - start)

    #######################################
    ############# Evaluate
    #######################################
    def evaluate_param(models, loader,  param1, param2, lam):
        model_brightness, model_hydration = models
        model_brightness = model_brightness.cuda()
        model_hydration = model_hydration.cuda()

        model_brightness.eval()
        model_hydration.eval()
        losses = []
        correct = 0
        total = 0
        param = ((1 - lam) * param1 + lam * param2) % 360
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs_brightness = TF.adjust_brightness(inputs, brightness_factor=param).cuda()
                inputs_hydration = inputs.cuda()  # No transformation for hydration

                outputs_brightness = model_brightness(inputs_brightness)
                outputs_hydration = model_hydration(inputs_hydration)

                outputs = (outputs_brightness + outputs_hydration) / 2  # Ensemble averaging

                pred = outputs.argmax(dim=1)
                correct += (labels.cuda() == pred).sum().item()
                total += len(labels)
                loss = F.cross_entropy(outputs, labels.cuda())
                losses.append(loss.item())
        return correct / total, np.array(losses).mean()

    print('LinearConnect:')

    linearconnect = []

    eval_dataset_list = ['test', 'valid_half_hydrated', 'valid_dry']
    t_brightness = np.linspace(0.5, 1.5, num=12, endpoint=True)
    
    
    for t, evdts_name in zip([0, 0.5, 1.0], eval_dataset_list):
        for t1 in t_brightness:
            mtmp_brightness, _ = utils.prepare_model(args, nchannels, nclasses)
            mtmp_hydration, _ = utils.prepare_model(args, nchannels, nclasses)
            mtmp_sd_brightness = {k: (((1-t1) * m_brightness0.state_dict()[k].cpu() + t1 * m_brightness1.state_dict()[k].cpu()) ) for k in m_brightness0.state_dict().keys()}
            mtmp_sd_hydration = {k: (((1-t) * m_hydration0.state_dict()[k].cpu() + t * m_hydration1.state_dict()[k].cpu()) ) for k in m_hydration0.state_dict().keys()}
            mtmp_brightness.load_state_dict(mtmp_sd_brightness)
            mtmp_hydration.load_state_dict(mtmp_sd_hydration)
            mtmp_brightness.to(device)
            mtmp_hydration.to(device)
    
            acc, loss = evaluate_param((mtmp_brightness, mtmp_hydration), loader=hydration_dataloaders[evdts_name],
                                       param1=params[0], param2=params[1], lam=t1)
    
            remaining_dts = [elem for elem in eval_dataset_list if elem != evdts_name]
    
            acc1, loss1 = evaluate_param((mtmp_brightness, mtmp_hydration), loader=hydration_dataloaders[remaining_dts[0]],
                                         param1=params[0], param2=params[1], lam=t1)
            
            acc2, loss2 = evaluate_param((mtmp_brightness, mtmp_hydration), loader=hydration_dataloaders[remaining_dts[1]],
                                         param1=params[0], param2=params[1], lam=t1)
    
            print(f"{t}, {t1}--> {[acc, acc1, acc2]}")
            linearconnect.append([t, acc, acc1, acc2])

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stats = {'linearconnect': linearconnect}
    np.save(f'{destination_name}/acc.npy', pickle.dumps(stats))

if __name__ == '__main__':
    main()
