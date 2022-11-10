import pandas as pd
import json


def compute_basics_analysis_df(title, df):
    title_str = " "+title+" "
    print(f'{title_str.upper().center(50, "-")}')
    print(f"Nombre de commentaires: {len(df)}")
    if title != "test":
        print(f'Moyenne des notes = {df["note"].mean():.3f}')
        print(f'Ecart type = {df["note"].std():.3f}')
    counts = df['movie'].value_counts()
    print(f'Moyenne du nombre de commentaires par film = {counts.mean():.3f}')
    print(f"Ecart type = {counts.std():.3f}")
    counts = df['user_id'].value_counts()
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
    df["note"] = df["note"].apply(make_float)
    return df.dropna()

df_dev = load_xml("dataset/dev.xml")


print(df_dev.keys())
print(df_dev.dtypes)
print(df_dev)
compute_basics_analysis_df("dev", df_dev)
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
df_test = load_xml("dataset/test.xml")
# compute_basics_analysis_df("test", df_test)
