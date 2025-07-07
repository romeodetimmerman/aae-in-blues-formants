import numpy as np
from sklearn.linear_model import LinearRegression


def compute_euclidean_distance_and_deltas(token, t20, t80):
    """
    compute euclidean distance and relative deltas between time points

    params
    ------
    token: pd.dataframe
        dataframe containing measurements for a single vowel token
    t20: float
        20% time point
    t80: float
        80% time point

    returns
    -------
    dict
        dictionary containing euclid_dist, f1_delta, and f2_delta
    """
    row = {}

    try:
        # find closest indices to t20 and t80
        idx_20 = (token["measurement_time"] - t20).abs().idxmin()
        idx_80 = (token["measurement_time"] - t80).abs().idxmin()

        # get formant values at these time points
        f1p_20, f2p_20 = token.loc[idx_20, "f1p"], token.loc[idx_20, "f2p"]
        f1p_80, f2p_80 = token.loc[idx_80, "f1p"], token.loc[idx_80, "f2p"]

        # compute euclidean distance
        row["euclid_dist"] = np.sqrt((f1p_80 - f1p_20) ** 2 + (f2p_80 - f2p_20) ** 2)

        # compute relative deltas
        epsilon = 1e-3

        row["f1_delta"] = (f1p_80 - f1p_20) / (
            (abs(f1p_80) + abs(f1p_20)) / 2 + epsilon
        )
        row["f2_delta"] = (f2p_80 - f2p_20) / (
            (abs(f2p_80) + abs(f2p_20)) / 2 + epsilon
        )

    except Exception:
        row["euclid_dist"] = row["f1_delta"] = row["f2_delta"] = np.nan

    return row


def compute_linear_slopes(token_middle):
    """
    compute linear slopes for f1 and f2 in middle portion

    params
    ------
    token_middle: pd.dataframe
        dataframe containing middle portion of vowel token

    returns
    -------
    dict
        dictionary containing f1_linear_slope and f2_linear_slope
    """
    row = {}

    try:
        if len(token_middle) >= 2:
            X = token_middle[["measurement_time"]].values
            y1 = token_middle["f1p"].values
            y2 = token_middle["f2p"].values

            # fit linear models
            model1 = LinearRegression().fit(X, y1)
            model2 = LinearRegression().fit(X, y2)

            row["f1_linear_slope"] = model1.coef_[0]
            row["f2_linear_slope"] = model2.coef_[0]
        else:
            row["f1_linear_slope"] = row["f2_linear_slope"] = np.nan

    except Exception:
        row["f1_linear_slope"] = row["f2_linear_slope"] = np.nan

    return row
