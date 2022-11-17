import pandas as pd
import json
import matplotlib.pyplot as plt


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


def boxplot_column(df, column="note"):
    boxplot = df.boxplot(column=column)
    plt.show()


def hist_column(df, column="note"):
    hist = df.hist(column=column, bins=9)
    plt.show()

def hist_mean_rate(df, column="user_id"):
    new_df = df.groupby([column]).note.mean()
    hist = new_df.hist(bins=9)
    plt.title("mean rating by "+column)
    plt.show()
