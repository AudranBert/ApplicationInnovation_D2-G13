import camembert_model as cm
import custom_dataset as cd
import to_csv as dm
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

logging.basicConfig(level=logging.INFO)
device="cuda" if torch.cuda.is_available() else "cpu"

def valid(model, valid_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
    test_loss /= len(valid_loader.dataset)

if __name__ == '__main__':
    logging.info("program is starting")
    train_dataset = cd.CustomDataset("dev")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    model = cm.CamembertCustom()
    valid(model, train_loader)
    logging.info("program end")






