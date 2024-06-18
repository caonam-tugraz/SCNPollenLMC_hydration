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
# import wandb
from torchvision import datasets, models, transforms
import torch.utils.data as data


os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='SCN Pollen')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlpb')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--dimensions', default=1, type=int)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()

# dataset_root_folder = './data/images_3_types_dry_7030'
dataset_root_folder = './data/images_3_types_dry_7030'

train_directory = f'{dataset_root_folder}_train'
valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'


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
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
}


# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=args.batchsize, shuffle=True),
    'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}

# fixed_params = [0.2, 0.5, 1.0, 1.5, 2.0]
fixed_params = [1.0]

def main():
    utils.set_seed(13)
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(dataset['train'].classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## download datasets
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)  # prints [batch_size, 1, 28, 28] and [batch_size]
    #     break  # remove break to go through all batches

    result_dict = {}
    for fixed_param in fixed_params:
        ######## prepare model structure
        model, save_dir = utils.prepare_model(args, nchannels, nclasses)
        # wandb.init(project=f"SCNPollen", entity="ahinea", name=f"One4One_{args.transform}_{save_dir}_{fixed_param}")
        model.to(device)
        print(model)
        print(utils.count_model_parameters(model))

        ######## train model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        def train(dataloader, model, loss_fn, optimizer, param):
            for batch, (X, y) in enumerate(tqdm(dataloader, desc='Training')):
                X, y = X.to(device), y.to(device)
                
                # if args.transform == "brightness":
                #     X = TF.adjust_brightness(X, brightness_factor=param)
                # elif args.transform == "contrast":
                #     X = TF.adjust_contrast(X, contrast_factor=param)
                # elif args.transform == "saturation":
                #     X = TF.adjust_saturation(X, saturation_factor=param)
                # elif args.transform == "sharpness":
                #     X = TF.adjust_sharpness(X, sharpness_factor=param)

                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        def validate(dataloader, model, loss_fn, param):
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    # if args.transform == "brightness":
                    #     X = TF.adjust_brightness(X, brightness_factor=param)
                    # elif args.transform == "contrast":
                    #     X = TF.adjust_contrast(X, contrast_factor=param)
                    # elif args.transform == "saturation":
                    #     X = TF.adjust_saturation(X, saturation_factor=param)
                    # elif args.transform == "sharpness":
                    #     X = TF.adjust_sharpness(X, sharpness_factor=param)

                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= len(dataloader)
            correct /= len(dataloader.dataset)
            print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
            return correct, test_loss

        for t in range(args.epochs):
            print(f"=================\n Epoch: {t + 1} \n=================")
            train(train_loader, model, loss_fn, optimizer, fixed_param)
            test_acc, test_loss = validate(test_loader, model, loss_fn, fixed_param)
            # wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        print("Done!")

        result_dict[str(fixed_param)] = test_acc
        # wandb.finish()
        
    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/One4One/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(result_dict))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
