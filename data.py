import pandas as pd


def load_data(path):
    return pd.read_csv(path)

def add_new_columns(df):
    dic = {0: "spring", 1: "summer", 2: "fall", 3: "winter"}
    df["season_name"] = df["season"].apply(lambda x: dic[x])
    #3
    df["Hour"] = df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).hour)
    df["Day"] = df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).day)
    df["Month"] = df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).month)
    df["Year"] = df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).year)
    #4
    df["is_weekend_holiday"] = 1 + df["is_weekend"].apply(lambda x: 1 if x else 0) + 2* df["is_holiday"].apply(lambda x: 1 if x else 0)
    #5
    df["t_diff"] = df["t2"] - df["t1"]
    return df
