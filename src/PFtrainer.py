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



from dataloaders.FullLoaderBubbleML import FullLoaderBubbleML, get_dataloader, get_dataset

"""


#TODO implement temporal bundling and PF method
#TODO also make classes instead of scripts with functions


"""


def load_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config

def validate(args, model, dataloader, criterion, device):
    # we calculate both timestep losses and full unrolling losses
    model.eval()
    
    steps = [t for t in range(args.tw, args.t_res-args.tw + 1)]
    val_loss_timestep = val_timestep(model=model,
                                    steps=steps,
                                    batch_size=args.batch_size,
                                    loader=dataloader,
                                    criterion=criterion,
                                    device=device)

    val_loss_unrolled = val_unrolled(model=model,
                                    steps=steps,
                                    batch_size=args.batch_size,
                                    dataloader=dataloader,
                                    criterion=criterion,
                                    device=device)

    return val_loss_timestep, val_loss_unrolled

def val_timestep(model, steps, batch_size, loader, graph_creator, criterion, device):
    for step in steps:
    
        #if (step != graph_creator.tw and step % graph_creator.tw != 0):
        #    continue

        losses = []
        for raw_data in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = create_data(args, raw_data, same_steps)
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        #print(f'Step {step}, mean loss {torch.mean(losses)}')
    return torch.mean(losses)

def val_unrolled(model, steps, batch_size, nr_gt_steps, nx_base_resolution, dataloader, criterion, device):
    losses = []
    for raw_data in dataloader:
        losses_tmp = []
        with torch.no_grad():
            #same_steps = [args.tw * nr_gt_steps] * batch_size
            same_steps = [step] * batch_size
            data, labels = create_data(args, raw_data, same_steps)
        
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            loss = criterion(pred, labels)

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(args.tw * (nr_gt_steps + 1), args.t_res - args.tw + 1, args.tw):
                same_steps = [step] * batch_size
                _, labels = args.create_data(args, raw_data, same_steps)
                
                labels = labels.to(device)
                pred = model(pred)
                loss = criterion(pred, labels) / nx_base_resolution
                losses_tmp.append(loss / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        #losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    #losses_base = torch.stack(losses_base)
    #print(f'Unrolled forward losses {torch.mean(losses)}')
    #print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return torch.mean(losses)

def train(args, epoch, model, dataloader, optimizer, criterion, device):

    max_unrolling = epoch if epoch <= args.max_unrolling else args.max_unrolling
    #print("max_unrolling: ", max_unrolling)
    unrolling = [r for r in range(max_unrolling + 1)]
    epoch_losses = []

    for i in range(args.tres):
        batch_losses = train_one_epoch(args, model, unrolling, args.batch_size, optimizer, dataloader, criterion, device)
        epoch_losses.append(batch_losses.mean().item())
    return epoch_losses

def train_one_epoch(args, model, unrolling, batch_size, optimizer, dataloader, criterion, device):
    
    losses = []
    for raw_data in dataloader:
        #print(raw_data.shape)
        
        unrolled_choice = random.choice(unrolling)
        #print("unrolled_unrolling: ", unrolled_choice)
        steps = [t for t in range(args.tw, args.tres - args.tw - (args.tw * unrolled_choice) + 1)]
        #print("steps: ", steps)
        random_steps = random.choices(steps, k=batch_size)
        #print("random_steps: ", random_steps)
        data, labels = create_data(args, raw_data, random_steps)
        #print('\nget data\n')
        data, labels = data.to(device), labels.to(device)
        
        with torch.no_grad():
            for _ in range(unrolled_choice):
                random_steps = [rs + args.tw for rs in random_steps]
                _, labels = create_data(args, raw_data, random_steps)
                #print(labels.shape)
                data = model(data)
                labels = labels.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, labels)

        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()
        
        losses.append(torch.Tensor(1))
    losses = torch.stack(losses)
    return losses

def create_data(args, raw_data, random_steps):
    data = torch.Tensor()
    labels = torch.Tensor()
    for (dp, step) in zip(raw_data, random_steps):
        print(raw_data.shape)
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
    args.wandb = config['wandb'] == "True"
    args.modelname = config['modelname']
    args.learning_rate = config['training']['learning_rate']

    

    if args.modelname == "UNet2D_DS2015":
        from modelComp.UNet import UNet2D
        model = UNet2D(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(args.device)
    else:
        raise ValueError('MODEL NOT RECOGNIZED')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = None
    

    train_files = [config['data_path'] + file for file in config['training']['files']]
    val_files = [config['data_path'] + file for file in config['validation']['files']]
    train_dataset = get_dataset(train_files)
    val_dataset = get_dataset(val_files)
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=1, shuffle=False)
    args.tres = train_dataset.tres
    

    # check that max unrolling steps times time window do not pass the t resolution ( unrolling_choice = 0 equals no pushforward)
    assert(args.tw * (args.max_unrolling + 2) <= args.tres)
    
    xres = 14
    yres = 15
    dt = 1

    if args.wandb:
        wandb.init(project="BubbleML_DS", name=args.modelname)
        wandb.config.update(config)

    for epoch in range(args.epochs):

        train_losses = train(args, epoch, model, train_loader, optimizer, criterion, device=args.device)
        # also validation
        print('validating...')
        val_loss_timestep, val_loss_unrolled = validate(args, model, val_loader, criterion, device=args.device)

        if epoch % 10 == 0:
            if val_loss_timestep < best_val_loss_timestep:
                best_val_loss_timestep = val_loss_timestep
                best_model_path = f"models/best_val_loss_timestep_{args.modelname}_E{epoch}.pth"
                torch.save(model.state_dict(), best_model_path)
            if val_loss_unrolled < best_val_loss_unrolled:
                best_val_loss_unrolled = val_loss_unrolled
                best_model_path = f"models/best_val_loss_unrolled_{args.modelname}_E{epoch}.pth"
                torch.save(model.state_dict(), best_model_path)

        if args.wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss_mean": sum(train_losses) / len(train_losses),
                        "val_loss_timestep": val_loss_timestep.item(),
                        "val_loss_unrolled": val_loss_unrolled.item(),
                        "train_losses": train_losses 
                    })
        
        print(f"Epoch {epoch}: Train Loss Mean = {sum(train_losses) / len(train_losses)}, "
                      f"Val Loss Timestep = {val_loss_timestep}, Val Loss Unrolled = {val_loss_unrolled}")
        
    if wandb:
        wandb.finish()
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
    