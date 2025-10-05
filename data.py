import pandas as pd


def load_data(path):
    return pd.read_csv(path)

def add_new_columns(df):
    dic = {0: "spring", 1: "summer", 2: "fall", 3: "winter"}
    df["season_name"] = df["season"].apply(lambda x: dic[x])

    return df
