import pandas as pd
from sklearn.linear_model import LinearRegression

"""
this script classifies PRICE vowels as monophtongs or diphthongs 
based on linear regression slopes for f1 and f2 measurements
"""


def compute_slopes(
    token,
    f1_column,
    f2_column,
    time_column="time_ms",
):
    """
    fit linear regression models to f1 and f2 measurements and return their respective slopes

    params
    ------
    token: pandas grouped dataframe
           measurements for a particular token
    f1_column: str
               column containing f1 measurements
    f2_column: str
               column containing f2 measurements
    time_column: str
               column containing time measurements (in milliseconds)

    returns
    -------
    slope_f1, slope_f2: pandas series
                        slopes for f1 and f2 for the given token (unit = Hz/ms)
    """
    # sort by time (likely redundant, but just to be sure)
    token = token.sort_values(time_column)

    # X is time_column as a 2D array for sklearn
    X = token[[time_column]].values

    # fit f1 slope
    y_f1 = token[f1_column].values
    model_f1 = LinearRegression()
    model_f1.fit(X, y_f1)
    slope_f1 = model_f1.coef_[0]

    # fit f2 slope
    y_f2 = token[f2_column].values
    model_f2 = LinearRegression()
    model_f2.fit(X, y_f2)
    slope_f2 = model_f2.coef_[0]

    return pd.Series({"slope_f1": slope_f1, "slope_f2": slope_f2})


def classify_tokens(row, threshold_f1, threshold_f2):
    """
    classify tokens based on f1 and f2 slopes

    params
    ------
    row: pandas series
    threshold_f1: int
                  global variable specifying cutoff for f1 slope
    threshold_f2: int
                  global variable specifying cutoff for f2 slope

    returns
    -------
    classification: bool
                    monophthong or diphthong

    """
    if row["slope_f1"] < threshold_f1 and row["slope_f2"] > threshold_f2:
        return "diphthong"
    else:
        return "monophthong"
