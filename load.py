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
        return float(v)
    except:
        return pd.np.nan

file = "dataset/dev.xml"
df_dev = pd.read_xml(file)

df_dev["note"] = df_dev["note"].apply(make_float)
# json_result = df.to_json()
# out = open("test.json", "w")
# json.dump(result, "test.csv")

# file = "dataset/train.xml"
# df_train = pd.read_xml(file)

# file = "dataset/test.xml"
# df_test = pd.read_xml(file)

print(df_dev.keys())
print(df_dev.dtypes)
print(df_dev)
# compute_basics_on_file("train",df_train)
compute_basics_on_file("dev", df_dev)
# compute_basics_on_file("test",df_test)
