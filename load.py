import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json
import matplotlib.pyplot as plt
import re
import pickle
from tqdm import tqdm
from collections import defaultdict
import analyzer as an
import statistics
split_char = "\W+"
regexed = re.compile(split_char)

#return une hashmap qui decrit le lexique
def extract_lexique(df):
    words = set()
    lexique = {}
    with tqdm(total=len(df)) as pbar:
        for idx, row in df.iterrows():
            sentence = row['commentaire'].lower()
            sentence = re.split(split_char, sentence)
            for j in sentence:
                if j not in words and j != "":
                    lexique[j] = 1
                    words.add(j)
                elif j in words:
                    lexique[j] = lexique[j]+1
            pbar.update(1)
    print("There is", len(lexique), "words")
    save_object(lexique,"lexique.obj")
    return(lexique)

def comment_average_char(df):
    return(df[['commentaire','note']].groupby(['note']).apply(len))

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

#df_dev = load_object("df_dev")
#df_lol = special_stuff(df_dev)

#final_dict = defaultdict(lambda:[])
#for key in df_lol:
    #print(value)
#    final_dict[key].append(statistics.fmean(df_lol[key]))
    #print(len(df_lol[key]))
#    final_dict[key].append(len(df_lol[key]))
    #final_dict[key].append(len(value))

#print(final_dict)




#df_mdr = pd.DataFrame.from_dict(final_dict,orient='index')
#save_object(df_mdr,'df_mdr')

df_mdr = load_object('df_mdr')
#fig, ax = plt.subplots()
#colormap = cm.viridis
#colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(df['l']))]

#for i,c in enumerate(colorlist):

#    x = df_mdr['0'][i]
#    y = df_mdr['1'][i]

#    ax.scatter(x, y, label=l, s=50, linewidth=0.1, c=c)

#ax.legend()

#plt.show()
#special_stuff(df_dev)
#print(df_dev.keys())
#print(df_dev.dtypes)
#print(df_dev)
#an.compute_basics_analysis_df("dev", df_dev)
#an.hist_mean_rate(df_dev, "user_id")
#an.hist_column(df_dev, "note")
#an.hist_mean_rate(df_dev, "movie")

# list_avg = comment_average_char(df_dev)
# print(list_avg)

# boxplot_column(df_dev, "note")
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
#df_test = load_xml("dataset/test.xml")
# compute_basics_analysis_df("test", df_test)
