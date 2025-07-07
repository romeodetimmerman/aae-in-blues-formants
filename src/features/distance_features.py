import numpy as np


def compute_euclidean_distance_and_deltas(token_middle, t20, t80):
    """
    compute euclidean distance and relative deltas between time points

    params
    ------
    token_middle: pd.dataframe
        dataframe containing middle portion of vowel token
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
        idx_20 = (token_middle["measurement_time"] - t20).abs().idxmin()
        idx_80 = (token_middle["measurement_time"] - t80).abs().idxmin()

        # get formant values at these time points
        f1p_20, f2p_20 = (
            token_middle.loc[idx_20, "f1p"],
            token_middle.loc[idx_20, "f2p"],
        )
        f1p_80, f2p_80 = (
            token_middle.loc[idx_80, "f1p"],
            token_middle.loc[idx_80, "f2p"],
        )

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
