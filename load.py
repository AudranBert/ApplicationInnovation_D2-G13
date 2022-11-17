import pandas as pd
import json
import matplotlib.pyplot as plt
import re
import pickle
from tqdm import tqdm
split_char = "\W+"
def compute_basics_analysis_df(title, df):
    title_str = " "+title+" "
    print(f'{title_str.upper().center(50, "-")}')
    print(f"Nombre de commentaires: {len(df)}")
    if title != "test":
        print(f'Moyenne des notes = {df["note"].mean():.3f}')
        print(f'Ecart type = {df["note"].std():.3f}')
    counts = df['movie'].value_counts()
    print(f'Nombre de films = {len(counts)}')
    print(f'Moyenne du nombre de commentaires par film = {counts.mean():.3f}')
    print(f"Ecart type = {counts.std():.3f}")
    counts = df['user_id'].value_counts()
    print(f'Nombre de users = {len(counts)}')
    print(f'Moyenne du nombre de commentaires par user = {counts.mean():.3f}')
    print(f"Ecart type = {counts.std():.3f}")


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
    return(df['commentaire'].apply(len).mean())

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename,'rb') as intp:
        return pickle.load(intp)


#df_dev = load_xml("dataset/dev.xml")


#print(df_dev.keys())
#print(df_dev.dtypes)
#print(df_dev)
#list_avg = comment_average_char(df_dev)
#print(list_avg)
#extract_lexique(df_dev)
lexique = load_object("lexique.obj")
for w in sorted(lexique,key=lexique.get,reverse=True):
    lexique[w]=lexique[w]-1
    if(lexique[w]>1000):
        print(w,lexique[w])

#print(len(lexique))
#compute_basics_analysis_df("dev", df_dev)
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
#df_test = load_xml("dataset/test.xml")
#compute_basics_analysis_df("test", df_test)
