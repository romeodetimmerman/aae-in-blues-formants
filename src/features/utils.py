import pandas as pd
import numpy as np


def normalize_speaker_formants(df, speaker_col="artist", formant_cols=["f1p", "f2p"]):
    """
    z-score normalize formant columns per speaker

    params
    ------
    df: pd.dataframe
        dataframe to normalize
    speaker_col: str
        name of the column containing the speaker id
    formant_cols: list[str]
        list of columns to normalize

    returns
    -------
    pd.dataframe
        dataframe with the normalized columns
    """
    df = df.copy()
    for speaker, group in df.groupby(speaker_col):
        for col in formant_cols:
            mean = group[col].mean()
            std = group[col].std()
            idx = group.index
            df.loc[idx, col] = (group[col] - mean) / std if std > 0 else 0
    return df
