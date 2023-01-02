

checkpoints_folder = "checkpoints"
pickle_folder = "pickle"    # joyeux
xml_folder = "dataset"      # joyeux
# checkpoints_folder = "../checkpoints"
# pickle_folder = "../pickle"
# xml_folder = "../dataset"
test_out_file = "test_predictions.pth"


import logging
import torch

logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_float(x):
    return float(x.replace(',', '.'))

def to_float_2(x):
    return (float(x.replace(',', '.'))*2)-1

def get_step(epoch, loader, batch_idx):
    return (epoch - 1) * len(loader) + batch_idx