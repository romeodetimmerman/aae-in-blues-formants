import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
script to transform columns in raw data
"""


def seconds_to_milliseconds(df, seconds_column):
    """
    transform seconds to milliseconds
    """
    return df[seconds_column] * 1000


def train_test(df, label_col="aae_realization"):
    """
    split data into training and test sets
    """
    return train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_col])


def calculate_relative_time_points(token, duration_column, time_points):
    """
    calculate absolute time values corresponding to relative time points (percentages)
    within a vowel token's duration.

    params
    ------
    token: pd.dataframe
        dataframe containing measurements for a single vowel token.
        must contain the duration column.
    duration_column: str
        name of the column containing the token's total duration (in ms).
        assumes duration is constant for all rows in the token.
    time_points: list[float]
        list of relative time points (percentages, e.g., [0.2, 0.8]).

    returns
    -------
    dict[str, float]
        dictionary mapping a string key (e.g., "t20", "t80") to the
        calculated absolute time value (in ms).
    """
    if token.empty or duration_column not in token.columns:
        # handle empty token or missing duration column gracefully
        return {f"t{int(p * 100)}": 0.0 for p in time_points}

    # assumes duration is the same for all rows of the token
    duration = token[duration_column].iloc[0]
    if pd.isna(duration) or duration <= 0:
        # handle invalid duration
        return {f"t{int(p * 100)}": 0.0 for p in time_points}

    time_values = {}
    for point in time_points:
        key = f"t{int(point * 100)}"  # create key like "t20", "t80"
        time_values[key] = point * duration

    return time_values


def normalize_speaker_formants(df, speaker_col="artist", formant_cols=["f1p", "f2p"]):
    """
    z-score normalize formant columns per speaker
    """
    df = df.copy()
    for speaker, group in df.groupby(speaker_col):
        for col in formant_cols:
            mean = group[col].mean()
            std = group[col].std()
            idx = group.index
            df.loc[idx, col] = (group[col] - mean) / std if std > 0 else 0
    return df
