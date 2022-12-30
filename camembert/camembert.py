import os
import time

import camembert_model as cm
import custom_dataset as cd
import to_csv as dm
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter()

def get_step(epoch, loader, batch_idx):
    return (epoch - 1) * len(loader.dataset) + batch_idx


def full_train_with_valid(nb_epoch=2):
    os.makedirs("checkpoints", exist_ok=True)
    model, optimizer, train_loader, valid_loader = init_training()
    criterion = nn.CrossEntropyLoss()
    best_valid_acc = 100
    best_valid_it = 0
    valid_epoch(0, model, valid_loader)
    for epoch in range(nb_epoch):  # train
        model.train()  # tell that the model will be trained
        t_loss = 0
        t_batch = 0
        for batch_idx, (x, target) in enumerate(tqdm(train_loader, desc=str(epoch) + ": Train batch")):  # for each batch
            model.train()
            x, target = x.to(device), target.to(device)  # send it to the device
            h = model.init_hidden(len(x)).data  # init to the size of
            model.zero_grad()  # init gradients to 0
            output, h = model(x, h)  # forward
            loss = criterion(output, target)  # compute the loss
            loss.backward()  # back propagation
            t_loss += loss
            t_batch += 1
            optimizer.step()  # gradient descent
            if get_step(epoch, train_loader, batch_idx) % 1000 == 0 and batch_idx != 0:
                size = len(target) * t_batch
                step_loss = t_loss / size
                writer.add_scalar('Loss/train', step_loss, get_step(epoch, train_loader, batch_idx))
                t_loss = 0
                t_batch = 0
            if get_step(epoch, train_loader, batch_idx) % 25000 == 0 and batch_idx != 0:
                torch.save(model, "checkpoints/last_model.pth")
                v_acc = valid_epoch(get_step(epoch, train_loader, batch_idx), model, valid_loader, during_epoch=True)
                if v_acc < best_valid_acc:  # keep the best weights
                    best_valid_acc = v_acc
                    torch.save(model, "checkpoints/best_model.pth")
        torch.save(model, "checkpoints/last_model.pth")
        v_acc = valid_epoch(epoch + 1, model, valid_loader)  # test validation
        if v_acc < best_valid_acc:  # keep the best weights
            best_valid_acc = v_acc
            torch.save(model, "checkpoints/best_model.pth")
    writer.close()
    test()



def train_epoch(epoch, model, optimizer, criterion, train_loader):
    model.train()  # tell that the model will be trained
    t_loss = 0
    t_batch = 0
    for batch_idx, (x, target) in enumerate(tqdm(train_loader, desc=str(epoch) + ": Train batch")):  # for each batch
        x, target = x.to(device), target.to(device)  # send it to the device
        h = model.init_hidden(len(x)).data  # init to the size of
        model.zero_grad()  # init gradients to 0
        output, h = model(x, h)  # forward
        loss = criterion(output, target)  # compute the loss
        loss.backward()  # back propagation
        t_loss += loss
        t_batch += 1
        optimizer.step()  # gradient descent
        if batch_idx % 1000 == 0 and batch_idx != 0:
            size = len(target) * t_batch
            step_loss = t_loss / size
            writer.add_scalar('Loss/train', step_loss, (epoch-1) * len(train_loader.dataset) + batch_idx)
            t_loss = 0
            t_batch = 0


def valid_epoch(epoch, model, valid_loader, during_epoch=False):
    model.eval()
    correct_tot = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(
            tqdm(valid_loader, desc=str(epoch) + ": Valid batch")):  # for each batch
            x, target = x.to(device), target.to(device)  # send it to the device
            h = model.init_hidden(len(x)).data  # init to the size of
            output, h = model(x, h)  # forward
            t = torch.argmax(target, dim=1).to(device).type(torch.float)
            pred = torch.argmax(output, dim=1).to(device).type(torch.float)
            correct_tot += pred.eq(t.view_as(pred)).sum()
    v_acc = correct_tot / len(valid_loader.dataset)
    if during_epoch:
        writer.add_scalar('Accuracy/valid_step', v_acc, epoch)
    else:
        writer.add_scalar('Accuracy/valid', v_acc, epoch)
    return v_acc


def test_epoch(model, test_loader):
    model.eval()
    model_predictions = []
    with torch.no_grad():
        for batch_idx, (x) in enumerate(tqdm(test_loader, desc="TEST batch")):  # for each batch
            x = x.to(device)  # send it to the device
            h = model.init_hidden(len(x)).data  # init to the size of
            output, h = model(x, h)  # forward
            pred = torch.argmax(output, dim=1).cpu().detach().type(torch.float)
            model_predictions.append(pred)
    predictions = torch.cat(model_predictions, dim=0)
    torch.save(predictions, "test_predictions.pth")

def test():
    test_dataset = cd.CustomDataset("test", note=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    model = torch.load("checkpoints/best_model.pth")  # load the best weights of the model and use to test
    time.sleep(0.01)
    test_epoch(model, test_loader)

def init_training():
    train_dataset = cd.CustomDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,
                              persistent_workers=True)
    valid_dataset = cd.CustomDataset("dev")
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    model = cm.CamembertCustom()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer, train_loader, valid_loader

def train():
    os.makedirs("checkpoints", exist_ok=True)
    model, optimizer, train_loader, valid_loader = init_training()
    best_valid_acc = 100
    best_valid_it = 0
    valid_epoch(0, model, valid_loader)
    nb_epoch = 2
    for epoch in range(nb_epoch):  # train
        train_epoch(epoch + 1, model, optimizer, nn.CrossEntropyLoss(), train_loader)
        v_acc = valid_epoch(epoch + 1, model, valid_loader)  # test validation
        if v_acc < best_valid_acc:  # keep the best weights
            best_valid_acc = v_acc
            best_valid_it = epoch + 1
            torch.save(model, "checkpoints/best_model.pth")
    writer.close()
    test()



if __name__ == '__main__':

    logging.info("program is starting")
    full_train_with_valid(10)
    # valid(model, train_loader)
    logging.info("program end")
