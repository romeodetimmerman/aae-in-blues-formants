import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import calculate_relative_time_points

"""
this script computes linear regression slopes for f1 and f2 measurements
"""


def compute_slopes(
    token: pd.DataFrame,
    f1_column: str,
    f2_column: str,
    time_column: str = "time_ms",
    duration_column: str = "duration_ms",
) -> pd.Series:
    """
    fit linear regression models to f1 and f2 measurements and return their respective slopes
    only using the middle 60% of the vowel duration (between 20% and 80% time points).

    params
    ------
    token: pandas dataframe
           measurements for a particular token
    f1_column: str
               column containing f1 measurements
    f2_column: str
               column containing f2 measurements
    time_column: str
               column containing time measurements (in milliseconds)
    duration_column: str
               column containing duration measurements

    returns
    -------
    slope_f1, slope_f2: pandas series
                        slopes for f1 and f2 for the given token (unit = hz/ms)
                        returns pd.na if fewer than 2 points in middle 60% or on error.
    """
    if token.empty or time_column not in token.columns:
        return pd.Series({"slope_f1": pd.NA, "slope_f2": pd.NA})

    # sort by time (likely redundant, but just to be sure)
    token = token.sort_values(time_column)

    # calculate 20% and 80% time points using the utility function
    time_targets = calculate_relative_time_points(
        token=token, duration_column=duration_column, time_points=[0.2, 0.8]
    )
    t20 = time_targets["t20"]
    t80 = time_targets["t80"]

    # check if time calculation was valid
    # if t20==t80 (e.g., zero duration), filtering might yield empty or single point df
    if pd.isna(t20) or pd.isna(t80):
        return pd.Series({"slope_f1": pd.NA, "slope_f2": pd.NA})

    # filter for middle 60% of vowel (measurements between t20 and t80 inclusive)
    token_middle = token[
        (token[time_column] >= t20) & (token[time_column] <= t80)
    ].copy()

    # check if enough data points for regression
    if len(token_middle) < 2:
        return pd.Series({"slope_f1": pd.NA, "slope_f2": pd.NA})

    # X is time_column as a 2D array for sklearn
    # ensure no NaN values in the columns used for regression
    token_middle.dropna(subset=[time_column, f1_column, f2_column], inplace=True)
    if len(token_middle) < 2:
        return pd.Series({"slope_f1": pd.NA, "slope_f2": pd.NA})

    X = token_middle[[time_column]].values

    # fit f1 slope
    try:
        y_f1 = token_middle[f1_column].values
        model_f1 = LinearRegression()
        model_f1.fit(X, y_f1)
        slope_f1 = model_f1.coef_[0]
    except Exception:
        slope_f1 = pd.NA  # return NA on any regression error

    # fit f2 slope
    try:
        y_f2 = token_middle[f2_column].values
        model_f2 = LinearRegression()
        model_f2.fit(X, y_f2)
        slope_f2 = model_f2.coef_[0]
    except Exception:
        slope_f2 = pd.NA  # return NA on any regression error

    return pd.Series({"slope_f1": slope_f1, "slope_f2": slope_f2})
