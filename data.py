import numpy as np
import pandas as pd
import datetime as dt


def load_data(path):
    return pd.read_csv(path)

def add_new_columns(df):
    #2
    dic = {0: "spring", 1: "summer", 2: "fall", 3: "winter"}
    df["season_name"] = df["season"].apply(lambda x: dic[x])
    #3
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")
    # now extract hour, date parts
    df["Hour"] = df["timestamp"].dt.hour
    df["Day"] = df["timestamp"].dt.day
    df["Month"] = df["timestamp"].dt.month
    df["Year"] = df["timestamp"].dt.year
    #drop now redundant timestamp column
    df = df.drop("timestamp", axis=1)
    #4
    df["is_weekend_holiday"] = 1 + df["is_weekend"].apply(lambda x: 1 if x else 0) + 2* df["is_holiday"].apply(lambda x: 1 if x else 0)
    #5
    df["t_diff"] = df["t2"] - df["t1"]
    return df

def data_analysis(df):
    #6
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr(numeric_only=True)
    print(corr.to_string())
    print()
    #7
    abs_corr = corr.abs()

    mask = np.triu(np.ones(abs_corr.shape, dtype=bool), k= 1)
    pairs_df = abs_corr.where(mask)
    corr_dict = {
        (row, col): pairs_df.loc[row, col]
        for row in pairs_df.index
        for col in pairs_df.columns
        if not pd.isna(pairs_df.loc[row,col])
    }
    top5 = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom5 = sorted(corr_dict.items(), key=lambda x: x[1])[:5]
    print("Highest correlated are: ")
    for ind, tuple in enumerate(top5):
        print(f"{ind+1}. {tuple[0]} with {tuple[1]:.6f}")
    print()
    print("Lowest correlated are: ")
    for ind, tuple in enumerate(bottom5):
        print(f"{ind+1}. {tuple[0]} with {tuple[1]:.6f}")
    print()
    avg_season_t_diff = df.groupby("season_name")["t_diff"].agg("mean")
    for (season, t_diff) in avg_season_t_diff.items():
        print(f"{season} average t_diff is {t_diff:.2f}")
    print(f"All average t_diff is {df["t_diff"].mean():.2f}")
    print()