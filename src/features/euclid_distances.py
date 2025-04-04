import pandas as pd
import numpy as np
from utils import calculate_relative_time_points

"""
this script computes the euclidean distance in the f1-f2 space
"""


def compute_distance(
    token: pd.DataFrame,
    f1_column: str,
    f2_column: str,
    duration_column: str = "duration_ms",
    time_column: str = "time_ms",
) -> pd.Series:
    """
    compute f1 and f2 at 20% and 80% mark of vowel
    and calculate euclidean distance between these two points

    params
    ------
    token: pandas dataframe
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
              returns pd.na if token is empty or time calculation fails
    """
    if token.empty or time_column not in token.columns:
        return pd.Series({"euclid_distance": pd.NA})

    # sort by time (likely redundant, but just to be sure)
    token = token.sort_values(time_column)

    # calculate 20% and 80% time points using the utility function
    time_targets = calculate_relative_time_points(
        token=token, duration_column=duration_column, time_points=[0.2, 0.8]
    )
    t20 = time_targets["t20"]
    t80 = time_targets["t80"]

    # check if time calculation was successful (might return 0.0 on error)
    if t20 == 0.0 and t80 == 0.0:
        # check if this is due to zero duration or empty token originally
        if token.empty or token[duration_column].iloc[0] <= 0:
            return pd.Series({"euclid_distance": pd.NA})
        # otherwise, it might be a valid case (e.g., start and end are the same)
        # proceed, but the distance will likely be 0 or nan

    # find indices of rows whose 'time' is closest to t20 and t80
    try:
        idx_20 = (token[time_column] - t20).abs().idxmin()
        idx_80 = (token[time_column] - t80).abs().idxmin()
    except ValueError:  # handle case where idxmin receives an empty series (e.g., if token becomes empty after filtering)
        return pd.Series({"euclid_distance": pd.NA})

    # extract F1, F2 at 20% and 80%
    f1_20, f2_20 = token.loc[idx_20, f1_column], token.loc[idx_20, f2_column]
    f1_80, f2_80 = token.loc[idx_80, f1_column], token.loc[idx_80, f2_column]

    # euclidean distance between (f1_20, f2_20) and (f1_80, f2_80)
    euclid_distance = np.sqrt((f1_20 - f1_80) ** 2 + (f2_20 - f2_80) ** 2)

    # return a series (so pandas can merge results across tokens)
    return pd.Series({"euclid_distance": euclid_distance})
