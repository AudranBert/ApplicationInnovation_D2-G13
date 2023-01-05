import os
import pickle

import pandas as pd

execution_id = "w_Upper"

checkpoints_folder = "checkpoints"
pickle_folder = "pickle"    # joyeux
xml_folder = "dataset"      # joyeux
export_folder = "export"
# checkpoints_folder = "../checkpoints"
# pickle_folder = "../pickle"
# xml_folder = "../dataset"
# export_folder = "../export"
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

def make_float(v):
    v = v.replace(",", ".")
    return float(v)

def load_xml(file_name):
    df = pd.read_xml(file_name)
    try:
        df["note"] = df["note"].apply(make_float)
    except:
        pass
    df.fillna('',inplace=True)
    return df.reset_index(drop=True)

def check_xml(pickle_file, data_file):
    if os.path.exists(pickle_file):
        df = load_object(pickle_file)
        print(f"Loading pickle: {pickle_file}")
    else:
        print(f"Loading xml: {data_file}")
        df = load_xml(data_file)
        save_object(df, pickle_file)
    return df

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

test_file = os.path.join(xml_folder, "test.xml")

def load_object(filename):
    with open(filename, 'rb') as intp:
        return pickle.load(intp)