import pandas as pd
import numpy as np
from utils import calculate_relative_time_points

####################
# relative delta f1 and f2 #
####################


def compute_deltas(
    vowel: pd.DataFrame,
    f1_column: str = "f1p",
    f2_column: str = "f2p",
    time_column: str = "time",
    duration_column: str = "duration",
) -> pd.Series:
    """
    compute the relative change in f1 and f2 between 20% and 80% of the vowel duration.

    params
    ------
    vowel: pd.dataframe
        dataframe containing measurements for a single vowel token.
    f1_column: str
        name of the column containing f1 measurements.
    f2_column: str
        name of the column containing f2 measurements.
    time_column: str
        name of the column containing time measurements.
    duration_column: str
        name of the column containing duration measurements.

    returns
    -------
    pd.series
        containing 'delta_f1' and 'delta_f2', the relative changes.
        returns nan if calculation is not possible (e.g., empty input, division by zero).
    """
    if vowel.empty or not all(
        col in vowel.columns
        for col in [time_column, duration_column, f1_column, f2_column]
    ):
        return pd.Series({"delta_f1": np.nan, "delta_f2": np.nan})

    # sort by time just in case
    vowel = vowel.sort_values(time_column)

    # calculate 20% and 80% time points using the utility function
    time_targets = calculate_relative_time_points(
        token=vowel,  # pass the dataframe to the function
        duration_column=duration_column,
        time_points=[0.2, 0.8],
    )
    t20 = time_targets["t20"]
    t80 = time_targets["t80"]

    # check for valid time points (utility returns 0.0 on error/invalid duration)
    # also check if the time column actually has values to compare against
    if (t20 == 0.0 and t80 == 0.0 and vowel[duration_column].iloc[0] <= 0) or vowel[
        time_column
    ].isnull().all():
        return pd.Series({"delta_f1": np.nan, "delta_f2": np.nan})

    # get the indices of the rows whose 'time' is closest to 20% and 80%
    try:
        idx_20 = (vowel[time_column] - t20).abs().idxmin()
        idx_80 = (vowel[time_column] - t80).abs().idxmin()
    except ValueError:  # handles cases where idxmin fails (e.g., all NaN time column)
        return pd.Series({"delta_f1": np.nan, "delta_f2": np.nan})

    # extract the f1p and f2p values for those two indices using parameter columns
    f1_20 = vowel.loc[idx_20, f1_column]
    f1_80 = vowel.loc[idx_80, f1_column]
    f2_20 = vowel.loc[idx_20, f2_column]
    f2_80 = vowel.loc[idx_80, f2_column]

    # compute deltas, avoiding division by zero
    delta_f1 = (f1_80 - f1_20) / f1_20 if f1_20 != 0 else np.nan
    delta_f2 = (f2_80 - f2_20) / f2_20 if f2_20 != 0 else np.nan

    # return a Series so pandas knows how to combine the results
    return pd.Series({"delta_f1": delta_f1, "delta_f2": delta_f2})
