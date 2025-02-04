import pandas as pd
import numpy as np

"""
this script classifies PRICE vowels as monophtongs or diphthongs 
based on the euclidean distance in the f1-f2 space
"""


def compute_distance(
    token,
    f1_column,
    f2_column,
    duration_column="duration",
    time_column="time",
):
    """
    compute f1 and f2 at 20% and 80% mark of vowel
    and calculate euclidean distance between these two points

    params
    ------
    token: pandas grouped dataframe
           measurements for a particular token
    f1_column: str
               column containing f1 measurements
    f2_column: str
               column containing f2 measurements
    duration_column: str
               column containing duration measurements
    time_column: str
               column containing time measurements

    returns
    -------
    distance: pandas series
              euclidean distance between 20% and 80% mark of vowel

    """
    duration = token[duration_column].iloc[0]
    t20 = 0.2 * duration
    t80 = 0.8 * duration

    # indices of rows whose 'time' is closest to 20% and 80%
    idx_20 = (token[time_column] - t20).abs().idxmin()
    idx_80 = (token[time_column] - t80).abs().idxmin()

    # extract F1, F2 at 20% and 80%
    f1_20, f2_20 = token.loc[idx_20, f1_column], token.loc[idx_20, f2_column]
    f1_80, f2_80 = token.loc[idx_80, f1_column], token.loc[idx_80, f2_column]

    # euclidean distance between (f1_20, f2_20) and (f1_80, f2_80)
    distance = np.sqrt((f1_20 - f1_80) ** 2 + (f2_20 - f2_80) ** 2)

    # return a series (so pandas can merge results across tokens)
    return pd.Series({"distance": distance})


def classify_vowels(row, threshold, distance_column="distance"):
    """
    classifify vowels based on euclidean distance

    params
    ------
    row: pandas series
    threshold: int
               global variable specifying cutoff for euclidean distance
    distance_column: str
                     column containing euclidean distance

    returns
    -------
    classification: bool
                    monophthong or diphthong
    """

    if row[distance_column] > threshold:
        return "diphthong"
    else:
        return "monophthong"
