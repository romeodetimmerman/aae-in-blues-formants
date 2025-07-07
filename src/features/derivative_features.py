import numpy as np
from scipy.interpolate import UnivariateSpline


def compute_derivative_stats(x, y, kind):
    """
    fit univariate spline and compute first/second derivative stats

    params
    ------
    x: np.array
        time values
    y: np.array
        formant values
    kind: str
        "first" or "second" derivative

    returns
    -------
    list
        list containing mean, min, max, std of the derivative
    """
    # verify kind
    if kind not in ["first", "second"]:
        raise ValueError("kind must be 'first' or 'second'")

    # check if there are enough points
    if len(x) < 5:
        return [np.nan] * 4

    # fit spline and compute derivative
    try:
        spline = UnivariateSpline(x, y, s=0.5)
        deriv = spline.derivative(n=1 if kind == "first" else 2)
        x_dense = np.linspace(np.min(x), np.max(x), 100)
        vals = deriv(x_dense)
        return [np.mean(vals), np.min(vals), np.max(vals), np.std(vals)]
    except Exception:
        return [np.nan] * 4


def compute_derivative_features(token_middle):
    """
    compute first and second derivative statistics for f1 and f2

    params
    ------
    token_middle: pd.dataframe
        dataframe containing middle portion of vowel token

    returns
    -------
    dict
        dictionary containing derivative statistics for f1 and f2
    """
    row = {}

    try:
        x = token_middle["measurement_time"].values
        y1 = token_middle["f1p"].values
        y2 = token_middle["f2p"].values

        # compute first and second derivative stats
        f1_first = compute_derivative_stats(x, y1, kind="first")
        f1_second = compute_derivative_stats(x, y1, kind="second")
        f2_first = compute_derivative_stats(x, y2, kind="first")
        f2_second = compute_derivative_stats(x, y2, kind="second")

        # unpack first derivative stats for f1
        (
            row["f1_mean_first_deriv"],
            row["f1_min_first_deriv"],
            row["f1_max_first_deriv"],
            row["f1_std_first_deriv"],
        ) = f1_first

        # unpack second derivative stats for f1
        (
            row["f1_mean_second_deriv"],
            row["f1_min_second_deriv"],
            row["f1_max_second_deriv"],
            row["f1_std_second_deriv"],
        ) = f1_second

        # unpack first derivative stats for f2
        (
            row["f2_mean_first_deriv"],
            row["f2_min_first_deriv"],
            row["f2_max_first_deriv"],
            row["f2_std_first_deriv"],
        ) = f2_first

        # unpack second derivative stats for f2
        (
            row["f2_mean_second_deriv"],
            row["f2_min_second_deriv"],
            row["f2_max_second_deriv"],
            row["f2_std_second_deriv"],
        ) = f2_second

    except Exception:
        # set all derivative features to nan if computation fails
        for prefix in ["f1", "f2"]:
            for stat in ["mean", "min", "max", "std"]:
                row[f"{prefix}_first_deriv_{stat}"] = np.nan
                row[f"{prefix}_second_deriv_{stat}"] = np.nan

    return row
