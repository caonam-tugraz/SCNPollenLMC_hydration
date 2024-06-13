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

import utils
import wandb

os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='Qcurves')
parser.add_argument('--dataset', default='FashionMNIST', type=str,
                    help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--arch', '-a', default='sconvb', type=str)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()


dataset_root_folder = f'./data/images_7_types_7030'
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
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
}

# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=args.batchsize, shuffle=True),
    'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}



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

    ######## prepare model structure
    params = [0.2, 2.0]
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
    for ep in range(args.epochs):
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            loss = 0
            for _ in range(10):
                t = random.random()
                param = ((1 - t) * params[0] + t * params[1]) % 360
                if args.transform == "brightness":
                    inputs_transformed = TF.adjust_brightness(inputs, brightness_factor=param)
                elif args.transform == "contrast":
                    inputs_transformed = TF.adjust_contrast(inputs, contrast_factor=param)
                elif args.transform == "saturation":
                    inputs_transformed = TF.adjust_saturation(inputs, saturation_factor=param)
                elif args.transform == "sharpness":
                    inputs_transformed = TF.adjust_sharpness(inputs, sharpness_factor=param)
                logits = m0.linearcurve(m0, m1, t, inputs_transformed)
                
                # print(logits)
                # print(labels)
                
                loss += lfn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {ep} loss: {loss}")
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
    def evaluate_param(model, loader, param1, param2, lam):
        model = model.cuda()
        param = ((1 - lam) * param1 + lam * param2) % 360
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                if args.transform == "brightness":
                    inputs = TF.adjust_brightness(inputs, brightness_factor=param)
                elif args.transform == "contrast":
                    inputs = TF.adjust_contrast(inputs, contrast_factor=param)
                elif args.transform == "saturation":
                    inputs = TF.adjust_saturation(inputs, saturation_factor=param)
                elif args.transform == "sharpness":
                    inputs = TF.adjust_sharpness(inputs, sharpness_factor=param)
                outputs = model(inputs.cuda())
                pred = outputs.argmax(dim=1)
                correct += (labels.cuda() == pred).sum().item()
                total += len(labels)
                loss = F.cross_entropy(outputs, labels.cuda())
                losses.append(loss.item())
        return correct / total, np.array(losses).mean()

    print('LinearConnect:')
    trange = np.linspace(0, 1.0, num=36, endpoint=True)
    linearconnect = []
    for t in trange:
        param = ((1 - t) * params[0] + t * params[1]) % 360
        mtmp, _ = utils.prepare_model(args, nchannels, nclasses)
        mtmp_sd = {k: (((1 - t) * m0.state_dict()[k].cpu() +
                          t * m1.state_dict()[k].cpu())
        ) for k in m0.state_dict().keys()}
        mtmp.load_state_dict(mtmp_sd)

        # run a single train epoch with augmentations to recalc stats
        mtmp.train()
        with torch.no_grad():
            for images, _ in train_loader:
                if args.transform == "brightness":
                    images = TF.adjust_brightness(images, brightness_factor=param)
                elif args.transform == "contrast":
                    images = TF.adjust_contrast(images, contrast_factor=param)
                elif args.transform == "saturation":
                    images = TF.adjust_saturation(images, saturation_factor=param)
                elif args.transform == "sharpness":
                    images = TF.adjust_sharpness(images, sharpness_factor=param)
                mtmp(images)
        mtmp.eval()

        acc, loss = evaluate_param(mtmp.cuda(), loader=test_loader, param1=params[0], param2=params[1], lam=t)
        acc0, loss0 = evaluate_param(mtmp.cuda(), loader=test_loader, param1=params[0], param2=params[1], lam=0)
        acc90, loss90 = evaluate_param(mtmp.cuda(), loader=test_loader, param1=params[0], param2=params[1], lam=0.5)
        acc180, loss180 = evaluate_param(mtmp.cuda(), loader=test_loader, param1=params[0], param2=params[1], lam=1.0)
        print(f"{param} --> {[acc, acc0, acc90, acc180]}")
        linearconnect.append([param, acc, acc0, acc90, acc180])

    stats = {'linearconnect': linearconnect}
    np.save(f'{destination_name}/acc.npy', pickle.dumps(stats))


if __name__ == '__main__':
    main()
