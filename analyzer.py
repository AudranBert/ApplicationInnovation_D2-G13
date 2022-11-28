import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def compute_basics_analysis_df(title, df):
    title_str = " " + title + " "
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
    print(f'Nombre de users uniques = {len(counts)}')
    print(f'Nombre de noms différents = {len(df.name.unique())}')
    print(f"Il y a {len(counts) - len(df.name.unique())} noms en double")
    print(f'Moyenne du nombre de commentaires par user = {counts.mean():.3f}')
    print(f"Ecart type = {counts.std():.3f}")
    print(f'{"-".upper().center(50, "-")}')


def violin_column(df, column="note", log=False):
    # fig, axes = plt.subplots()

    # new_df = df.groupby([column]).note.mean()
    # print(new_df)
    # axes.violinplot(dataset=[new_df.values])
    sns.violinplot(data=df, x="note", y=column,)
    plt.title(f"violin plot of mean rating by {column} {'with a log scale' if log else ''} ")
    if log:
        plt.yscale('log')
    plt.show()

def boxplot_column(df, column="note"):
    boxplot = df.boxplot(column=column)
    plt.title("box plot of"+column)
    plt.show()


def hist_column(df, column="note"):
    hist = df.hist(column=column, bins=9)
    plt.title("hist of "+column)
    plt.show()

def hist_mean_rate(df, column="user_id"):
    new_df = df.groupby([column]).note.mean()
    bins = [i / 2 for i in range(1, 10)]
    hist = new_df.hist(bins=bins)
    plt.title("mean rating by "+column)
    plt.show()
