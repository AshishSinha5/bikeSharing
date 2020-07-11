import pandas as pd


def mod_data(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(df['datetime'])
    df.drop(columns='datetime', inplace=True)
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df.reset_index(drop=True, inplace=True)
    return df


def min_max_normalize(df, cols, min, max):
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * (max - min) + min
    return df


def extract_date(df):
    return df['datetime']
