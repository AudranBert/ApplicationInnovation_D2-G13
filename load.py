import pandas as pd
import json
import matplotlib.pyplot as plt
import re
import pickle
from tqdm import tqdm
from collections import defaultdict
import statistics
import analyzer as an
split_char = "\W+"
errors = {'note_unvalid': 0}

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
    v = v.replace(",", ".")
    return float(v)

def find_nan(df):
    print(len(df))
    print(f"There is :\n{df.isnull().sum()} \nNaN rows")

def load_xml(file_name, print_infos=False):
    df = pd.read_xml(file_name)
    if print_infos:
        print(df.keys())
        print(df.dtypes)
        print(df)
        find_nan(df)
    try:
        df["note"] = df["note"].apply(make_float)
    except:
        pass
    return df.dropna()

def special_stuff(df):
    new_df = defaultdict(lambda: [])
    with tqdm(total=len(df)) as pbar:
        for idx,row in df.iterrows():
            sentence = row['commentaire'].lower()
            for c in sentence:
                #print(c)
                if (c.isalnum() == False) and (c != ' '):
                    new_df[c].append(row['note'])
            pbar.update(1)
    return(new_df)


def get_comment_properties(df):
    l = []
    with tqdm(total=len(df)) as pbar:
        for idx, row in df.iterrows():
            raw = row['commentaire']
            sentence_lower = row['commentaire'].lower()
            words = re.split(split_char, sentence_lower)
            l.append([row['movie'], row['review_id'], row['name'], row['user_id'], row['note'], len(raw), len(words), sum(1 for c in sentence_lower if c.islower()), sum(1 for c in raw if c.isupper())])
            pbar.update(1)
    return pd.DataFrame(l, columns=['movie', 'review_id', 'name', 'user_id', 'note', 'len_comment', 'len_wrd', 'len_letters', 'len_upper'])

df_mdr = load_object('df_mdr')
#df_dev = load_xml("dataset/dev.xml")
#save_object(df_dev,"df_mdr")
# an.compute_basics_analysis_df("dev", df_dev)
#print(df_mdr.head())
dict_emoji = special_stuff(df_mdr)
plt.figure(figsize=(20,8))
#print(df_emoji)
for emoji, notes_list in dict_emoji.items():
    plt.bar(emoji,sum(notes_list)/len(notes_list),align='edge',width=0.5)

plt.tight_layout()
plt.tick_params(axis="x",which='major',labelsize=7)

plt.show()
#df_comments = get_comment_properties(df_mdr)
#df_comments.plot(y="note",x="len_letters",alpha=0.5,kind="scatter")
#plt.show()

# an.hist_column(df_dev, "note")
# an.hist_mean_rate(df_dev, "user_id")
# an.violon_column(df_dev, 'user_id')
# an.hist_mean_rate(df_dev, "movie")
# an.violon_column(df_dev, 'movie')

# list_avg = comment_average_char(df_dev)
# print(list_avg)

# boxplot_column(df_dev, "note")
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
# df_test = load_xml("dataset/test.xml")
# compute_basics_analysis_df("test", df_test)
