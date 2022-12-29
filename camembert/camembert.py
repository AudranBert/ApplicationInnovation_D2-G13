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

def train_epoch(epoch,network,optimizer,train_loader,criterion):
    network.train()  # tell that the model will be trained
    t_loss = 0
    t_batch = 0
    for batch_idx, (x, target) in enumerate(tqdm(train_loader, desc=str(epoch)+": Train batch")):  # for each batch
        x = x.to(device)  # send it to the device
        target = target.to(device)
        h = network.init_hidden(len(x))  # init the layer with the size of the protein
        h = h.data
        network.zero_grad()  # init gradients to 0
        output, h = network(x, h)  # forward
        # # t = torch.argmax(output[:,-1,:],dim=1).float()
        # t = torch.argmax(output, dim=1).float()
        # print(output)
        loss = criterion(output, target)  # compute the loss
        loss.backward()  # back propagation
        t_loss += loss
        t_batch += 1
        optimizer.step()  # gradient descent
        if batch_idx % 250 == 10:
            size = len(target) * t_batch
            step_loss = t_loss/size
            writer.add_scalar('Loss/train', step_loss, batch_idx)
            t_loss = 0
            t_batch = 0


def valid_epoch(model, valid_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
    test_loss /= len(valid_loader.dataset)

def train():
    train_dataset = cd.CustomDataset("dev")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6,
                              persistent_workers=True)
    # valid_loader = DataLoader(validation_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=6,
    #                           persistent_workers=True)
    # test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    model = cm.CamembertCustom()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_epoch(0, model, optimizer, train_loader, nn.CrossEntropyLoss())

if __name__ == '__main__':
    logging.info("program is starting")
    train()
    # valid(model, train_loader)
    logging.info("program end")






