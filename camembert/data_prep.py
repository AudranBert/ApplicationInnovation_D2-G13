import os
import pickle
import pandas as pd
from transformers import CamembertTokenizer
import numpy as np
from params import *

tokenizer = CamembertTokenizer.from_pretrained("camembert-base", do_lower_case=True)



def str_to_token(x):
    if not x:
        x = "a"
        reducted_tokens = tokenizer.tokenize(x)
    else:
        tokenized_sentence = tokenizer.tokenize(x)
        reducted_tokens = tokenized_sentence[0:255] + tokenized_sentence[-256:-1]
    vectorized = tokenizer.encode(reducted_tokens)
    vectorized = np.asarray(vectorized, dtype=np.int32)
    vectorized = np.pad(vectorized, (0, 512-len(vectorized)), 'constant', constant_values=(0))
    # logging.info(vectorized)
    # if len(reducted_tokens) > 500:
    #     print(f"{len(tokenized_sentence[0:256])} et {len(tokenized_sentence[-257:-1])} = {len(reducted_tokens)}", flush=True)
    return vectorized



def to_tokens(comments):
    return comments.apply(str_to_token)


def pickle_to_csv(dataset, note=True):
    df = dataset_to_pickle(dataset, note)


def dataset_to_pickle(dataset, note=True):
    if not os.path.exists(os.path.join(pickle_folder, dataset+".p")):
        logging.info(f"Loading xml of: {dataset}")
        df_data = pd.read_xml(os.path.join(xml_folder, dataset + ".xml"))
        os.makedirs(pickle_folder, exist_ok=True)
        logging.info(f"Tokenization of: {dataset}")
        if note:
            df_token = pd.DataFrame(columns=['commentaire', 'note'])
            df_token['note'] = df_data['note'].apply(to_float)
            # df_token['note'] = to_float(df_data['note'])
        else:
            df_token = pd.DataFrame(columns=['commentaire'])
        df_token['commentaire'] = to_tokens(df_data['commentaire'])
        with open(os.path.join(pickle_folder, dataset + ".p"), 'wb') as f:
            pickle.dump(df_token, f)
        logging.info(f"Saving pickle: {dataset}")
    else:
        logging.info(f"Loading pickle: {dataset}")
        with open(os.path.join(pickle_folder, dataset+".p"), 'rb') as f:
            df_token = pickle.load(f)
    # df_token['commentaire'] = df_data['commentaire']
    # print(df_token)
    return df_token

