import pandas as pd
import json
import matplotlib.pyplot as plt
import re
import pickle
from tqdm import tqdm
import analyzer as an
split_char = "\W+"

#return une hashmap qui decrit le lexique
def extract_lexique(df):
    words = []
    lexique = {}
    with tqdm(total=len(df)) as pbar:
        for idx, row in df.iterrows():
            sentence = row['commentaire'].lower()
            sentence = re.split(split_char, sentence)
            for j in sentence:
                if j not in words and j != "":
                    lexique[j] = 1
                    words.append(j)
                elif j in words:
                    lexique[j] = lexique[j]+1
            pbar.update(1)
    print("There is", len(lexique), "words")
    save_object(lexique,"lexique.obj")
    return(lexique)

def comment_average_char(df):
    return(df[['commentaire','note']].groupby(['note']).apply(len))

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename,'rb') as intp:
        return pickle.load(intp)

def make_float(v):
    try:
        v = v.replace(",", ".")
        return float(v)
    except:
        return pd.np.nan

def load_xml(file_name):
    df = pd.read_xml(file_name)
    try:
        df["note"] = df["note"].apply(make_float)
    except:
        pass
    return df.dropna()

df_dev = load_xml("dataset/dev.xml")


print(df_dev.keys())
print(df_dev.dtypes)
print(df_dev)
an.compute_basics_analysis_df("dev", df_dev)
an.hist_mean_rate(df_dev, "user_id")
an.hist_column(df_dev, "note")
an.hist_mean_rate(df_dev, "movie")

# list_avg = comment_average_char(df_dev)
# print(list_avg)

# boxplot_column(df_dev, "note")
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
df_test = load_xml("dataset/test.xml")
# compute_basics_analysis_df("test", df_test)
