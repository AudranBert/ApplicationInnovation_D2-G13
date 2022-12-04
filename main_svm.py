import sys
import os
import pandas as pd
import re

from tqdm import tqdm

import pickle

pickle_folder = "pickle"
pickle_needed_folder = "pickle_needed"

data_folder = "dataset"
train_file = "train.xml"
test_file = "test.xml"
dev_file = "test.xml"
train_file = os.path.join(data_folder, train_file)
dev_file = os.path.join(data_folder, dev_file)
test_file = os.path.join(data_folder, test_file)


split_char = "\W+"


def make_float(v):
    v = v.replace(",", ".")
    return float(v)


def load_xml(file_name):
    df = pd.read_xml(file_name)
    try:
        df["note"] = df["note"].apply(make_float)
    except:
        pass
    df = df.dropna()
    return df.reset_index(drop=True)


def extract_lexique(all):
    # 2 et 3
    words = set()
    lexique = dict()
    ct = 1
    with tqdm(total=len(all)) as pbar:
        for idx, row in all.iterrows():
            sentence = row['commentaire'].lower()
            sentence = re.split(split_char, sentence)
            for j in sentence:
                if j not in words and j != "":
                    lexique[j] = ct
                    words.add(j)
                    ct += 1
            pbar.update(1)
    print("There is", len(lexique), "words")
    return lexique


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as intp:
        return pickle.load(intp)


def check_if_pickle_exist(pickle_file: str):
    if os.path.exists(pickle_file):
        return load_object(pickle_file)
    else:
        return False

def check_xml(pickle_file, data_file):
    if os.path.exists(pickle_file):
        df = load_object(pickle_file)
        print(f"Loading pickle: {pickle_file}")
    else:
        print(f"Loading xml: {data_file}")
        df = load_xml(data_file)
        save_object(df, pickle_file)
    return df



def get_occ_reviews(data):
    # 4
    occ = list()
    with tqdm(total=len(data)) as pbar:
        for idx, row in data.iterrows():
            occ.append(dict())
            sentence = row['commentaire'].lower()
            sentence = re.split(split_char, sentence)
            for j in sentence:
                if j not in occ[-1] and j != "":
                    occ[-1][j] = 1
                elif j != "":
                    occ[-1][j] += 1
            pbar.update(1)
    return occ

def convert_reviews(data, occ, mode="Train"):
    new_reviews = []
    with tqdm(total=len(data)) as pbar:
        for idx, row in data.iterrows():
            try:
                l = []
                if mode == "Test":
                    line = "1.0"
                else:
                    line = str(row['note'])
                for i in occ[idx]:
                    l.append([lexique[i], occ[idx][i]])
                l = sorted(l, key=lambda x: x[0])
                for i, j in l:
                    line = line + " " + str(i) + ":" + str(j)
                new_reviews.append(line)
            except IndexError as err:
                print(err)
                print(f"{len(data)} vs {len(occ)}")
                print(f"{idx}: {row}")
                # print(f"Last index of occ was {occ.last_index()}")
            pbar.update(1)
    return new_reviews

if __name__ == '__main__':

    # 1
    pickle_file = os.path.join(pickle_folder, "train.p")
    train = check_xml(pickle_file, train_file)

    pickle_file = os.path.join(pickle_folder, "dev.p")
    dev = check_xml(pickle_file, dev_file)

    pickle_file = os.path.join(pickle_folder, "test.p")
    test = check_xml(pickle_file, test_file)
    # test_f = pd.read_csv(file1, sep="\t", header=None, names=['sentence', 'sentiment'], skiprows=10, index_col=0)
    # test_f['file'] = "Test"
    # train_f = pd.read_csv(file2, sep="\t", header=None, names=['sentence', 'sentiment'], skiprows=10, index_col=0)
    # train_f['file'] = "Train"

    all = pd.concat([train, test], ignore_index=True)

    # all.to_csv("test.csv")
    # print(all)

    # print("There is ", len(train), "rows in train")
    # print("There is ", len(test), "rows in test")
    # print("There is ", len(all), "rows in total")




    pickle_file = os.path.join(pickle_folder, "lexique_all.p")
    if not os.path.exists(pickle_file):
        print(f"Extracting lexical...")
        lexique = extract_lexique(all)
        save_object(lexique, pickle_file)
    else:
        print(f"Loading pickle: {pickle_file}")
        lexique = load_object(pickle_file)

    pickle_file = os.path.join(pickle_folder, "occ_train.p")
    if not os.path.exists(pickle_file):
        print(f"Getting occurences...")
        occ = get_occ_reviews(train)
        save_object(occ, pickle_file)
    else:
        print(f"Loading pickle: {pickle_file}")
        occ = load_object(pickle_file)

    svm_file = "train.svm"
    if not os.path.exists(svm_file):
        train_tmp = convert_reviews(train, occ, "Train")
        df_train = pd.DataFrame(train_tmp)
        df_train.to_csv(svm_file, index=False, header=None)

    pickle_file = os.path.join(pickle_folder, "occ_test.p")
    if not os.path.exists(pickle_file):
        print(f"Getting occurences...")
        occ = get_occ_reviews(test)
        save_object(occ, pickle_file)
    else:
        print(f"Loading pickle: {pickle_file}")
        occ = load_object(pickle_file)

    svm_file = "test.svm"
    if not os.path.exists(svm_file):
        test_tmp = convert_reviews(test, occ, "Test")
        df_test = pd.DataFrame(test_tmp)
        df_test.to_csv(svm_file, index=False, header=None)
    # print(train)
    # df_test = pd.DataFrame(test)
    # df_test.to_csv("test.svm", index=False, header=None)
