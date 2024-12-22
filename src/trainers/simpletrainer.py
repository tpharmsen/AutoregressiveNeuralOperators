import torch
import torch.nn as nn
import torch.nn.functional as F


def one_epoch_train_simple(model, train_dataloader, optimizer, DEVICE):
    epoch_train_loss = []
    model.train()

    for iter, (input, label) in enumerate(train_dataloader):
        input = input.to(DEVICE).float()
        label = label.to(DEVICE).float()
        pred = model(input)
        optimizer.zero_grad()
        loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())

    avg_train_loss = torch.mean(torch.tensor(epoch_train_loss))
    return avg_train_loss

def one_epoch_val_simple(model, val_dataloader, DEVICE):
    epoch_val_loss = []
    model.eval()

    with torch.no_grad():
        for iter, (input, label) in enumerate(val_dataloader):
            input = input.to(DEVICE).float()
            label = label.to(DEVICE).float()
            pred = model(input)
            loss = F.mse_loss(pred, label)
            epoch_val_loss.append(loss.item())


    avg_val_loss = torch.mean(torch.tensor(epoch_val_loss))
    return avg_val_loss
