import os
import pickle
import pandas as pd
from transformers import CamembertTokenizer
import numpy as np
import string
from main_svm import *
from params import *
#from nltk.corpus import stopwords
#import re
tokenizer = CamembertTokenizer.from_pretrained("camembert-base", do_lower_case=True)
tokenizer = "hello"
#stopwords.words("french")
stopwords_list = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 
'étants', 'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 
'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent']
#regex_exp = re.compile(r'|'.join(stopwords_list),re.IGNORECASE)
#regex_exp = re.compile("|".join(stopwords_list))
def stopwords_remove(x):
    x.translate(x.maketrans('','',string.punctuation))
    splits = x.split()
    cleaned = ["" if x.lower() in stopwords_list else x for x in splits]
    cleaned = ' '.join(cleaned)
    return cleaned


def remove_stopwords(df):
    df["commentaire"] = df["commentaire"].apply(stopwords_remove)
    return df

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


if __name__ == '__main__':

    os.makedirs(pickle_folder, exist_ok=True)
    # 1
    pickle_file = os.path.join(pickle_folder, "train.p")
    train = check_xml(pickle_file, train_file)

    pickle_file = os.path.join(pickle_folder, "dev.p")
    dev = check_xml(pickle_file, dev_file)
    remove_stopwords(dev)