import pandas as pd

"""
script to transform columns in raw data
"""


def normalize_time(df, token_column, time_column):
    """
    normalize time column per
    """
    return df.groupby(token_column)[time_column].transform(lambda x: x / x.max())


def normalize_formants(df, formant_column):
    """
    normalize formants
    """
    # normalize f1p and f2p
    return (df[formant_column] - df[formant_column].mean()) / df[formant_column].std()


def seconds_to_milliseconds(df, seconds_column):
    """
    transform seconds to milliseconds
    """
    return df[seconds_column] * 1000
