import pandas as pd
import json


def compute_basics_on_file(title, df):
    title = " "+title+" "
    print(f'{title.upper().center(50, "-")}')
    print(f"Nombre de commentaires: {len(df)}")
    # mean = df["note"].mean()
    #print(f'{mean}')

def make_float(v):
    try:
        v=v.replace(",",".")
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
# compute_basics_on_file("train",df_train)
compute_basics_on_file("dev", df_dev)
# compute_basics_on_file("test",df_test)
