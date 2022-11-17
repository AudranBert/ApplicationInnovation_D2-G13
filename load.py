import pandas as pd
import json
import matplotlib.pyplot as plt
import analyzer as an



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


df_dev = load_xml("dataset/dev.xml")

print(df_dev.keys())
print(df_dev.dtypes)
print(df_dev)

an.compute_basics_analysis_df("dev", df_dev)
an.hist_mean_rate(df_dev, "user_id")
an.hist_column(df_dev, "note")
an.hist_mean_rate(df_dev, "movie")


# boxplot_column(df_dev, "note")
# df_train = load_xml("dataset/train.xml")
# compute_basics_analysis_df("train", df_train)
df_test = load_xml("dataset/test.xml")
# compute_basics_analysis_df("test", df_test)
