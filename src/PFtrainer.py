import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import wandb
import argparse
import yaml
import random

print('test')

from dataloaders.FullLoaderBubbleML import FullLoaderBubbleML, get_dataloader, get_dataset

print('test')
"""


#TODO implement temporal bundling and PF method
#TODO also make classes instead of scripts with functions


"""


def load_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config

def train(args, model, dataloader, optimizer, criterion, device):
    
    for epoch in range(args.epochs):

        max_unrolling = epoch if epoch <= args.max_unrolling else args.max_unrolling
        print("max_unrolling: ", max_unrolling)
        unrolling = [r for r in range(max_unrolling + 1)]

        for i in range(args.tres):
            train_one_epoch(args, model, unrolling, args.batch_size, optimizer, dataloader, criterion, device)

def train_one_epoch(args, model, unrolling, batch_size, optimizer, dataloader, criterion, device):
    
    losses = []
    for raw_data in dataloader:
        #optimizer.zero_grad()
        unrolled_choice = random.choice(unrolling)
        print("unrolled_unrolling: ", unrolled_choice)
        steps = [t for t in range(args.tw, args.tres - args.tw - (args.tw * unrolled_choice) + 1)]
        print("steps: ", steps)
        random_steps = random.choices(steps, k=batch_size)
        print("random_steps: ", random_steps)
        data, labels = create_data(args, raw_data, random_steps)
        print('\nget data\n')
        
        with torch.no_grad():
            for _ in range(unrolled_choice):
                random_steps = [rs + args.tw for rs in random_steps]
                _, labels = create_data(args, raw_data, random_steps)
            
                #data = model(data)
                #labels = labels.to(device)
        """
        pred = model(data)
        loss = criterion(pred, labels)

        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()
        """
        losses.append(torch.Tensor(1))
    losses = torch.stack(losses)
    return losses

def create_data(args, data, random_steps):
    data = torch.Tensor()
    labels = torch.Tensor()
    for (dp, step) in zip(data, random_steps):
        d = dp[step - args.tw:step]
        l = dp[step:args.tw + step]
        data = torch.cat((data, d[None, :]), 0)
        labels = torch.cat((labels, l[None, :]), 0)

    return data, labels


def main(args: argparse):
    config = load_config(args.conf)

    # check directories for saving...

    args.epochs = config['training']['epochs']
    args.epochs = 10
    args.device = torch.device(config["device"])
    args.max_unrolling = config['training']['max_unrolling']
    args.tw = config['tw']
    args.batch_size = 1 # config[]

    

    model = None
    optimizer = None
    criterion = None
    

    train_files = [config['data_path'] + file for file in config['training']['files']]
    #val_files = [config['data_path'] + file for file in config['validation']['files']]
    train_dataset = get_dataset(train_files)
    #val_dataset = get_dataset(val_files)
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #val_loader = get_dataloader(val_dataset, batch_size=1, shuffle=False)
    args.tres = train_dataset.tres
    args.tres = 20

    # check that max unrolling steps times time window do not pass the tres
    #assert(args.tw * args.max_unrolling  < args.tres)
    
    xres = 14
    yres = 15
    dt = 1

    train(args, model, train_loader, optimizer, criterion, device=args.device)


    print('ending...')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the BubbleML dataset')
    # add parse arguments
    print('starting...')
    # load config
    parser.add_argument("--conf", type=str, default="conf/example.yaml")
    # get data ready
    args = parser.parse_args()
    main(args)
    