import sys
import os
import pandas as pd
import re

file1 = "data_deft2017/task1-testGold.csv"
file2 = "data_deft2017/task1-train.csv"

split_char = "\W+"

if __name__ == '__main__':

    # 1
    test_f = pd.read_csv(file1, sep="\t", header=None, names=['sentence', 'sentiment'], skiprows=10, index_col=0)
    test_f['file'] = "Test"
    train_f = pd.read_csv(file2, sep="\t", header=None, names=['sentence', 'sentiment'], skiprows=10, index_col=0)
    train_f['file'] = "Train"
    all = pd.concat([train_f, test_f], ignore_index=True)

    # all.to_csv("test.csv")
    # print(all)

    print("There is ", len(train_f), "rows in train")
    print("There is ", len(test_f), "rows in test")
    print("There is ", len(all), "rows in total")

    # 2 et 3
    words = []
    lexique = {}
    occ = []
    ct = 1
    for idx, row in all.iterrows():
        sentence = row['sentence'].lower()
        sentence = re.split(split_char, sentence)
        for j in sentence:
            if j not in words and j != "":
                lexique[j] = ct
                words.append(j)
                ct += 1
    print("There is", len(lexique), "words")


    # 4
    occ = []
    for idx, row in all.iterrows():
        occ.append({})
        sentence = row['sentence'].lower()
        sentence = re.split(split_char, sentence)
        for j in sentence:
            if j not in occ[-1] and j != "":
                occ[-1][j] = 1
            elif j != "":
                occ[-1][j] += 1



    label = {"mixed": 3, "objective":0, "negative": 2, "positive":1 }

    train = []
    test = []
    for idx, row in all.iterrows():
        l = []
        line = str(label[row['sentiment']])
        for i in occ[idx]:
            l.append([lexique[i], occ[idx][i]])
        l = sorted(l, key=lambda x: x[0])
        for i, j in l:
            line = line + " " + str(i) + ":" + str(j)
        if row['file'] == "Test":
            test.append(line)
        elif row['file'] == "Train":
            train.append(line)

    # print(train)
    df_train = pd.DataFrame(train)
    df_train.to_csv("train.svm", index=False, header=None)
    df_test = pd.DataFrame(test)
    df_test.to_csv("test.svm", index=False, header=None)
