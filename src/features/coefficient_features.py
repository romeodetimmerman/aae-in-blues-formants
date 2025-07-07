import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline


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


def compute_quadratic_coefficients(token_middle):
    """
    compute quadratic coefficients for f1 and f2

    params
    ------
    token_middle: pd.dataframe
        dataframe containing middle portion of vowel token

    returns
    -------
    dict
        dictionary containing f1_quadratic_coef and f2_quadratic_coef
    """
    row = {}

    try:
        X = token_middle[["measurement_time"]].values
        y1 = token_middle["f1p"].values
        y2 = token_middle["f2p"].values

        # create polynomial features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # fit quadratic models
        model1 = LinearRegression().fit(X_poly, y1)
        model2 = LinearRegression().fit(X_poly, y2)

        # extract quadratic coefficients (index 2 for x^2 term)
        row["f1_quadratic_coef"] = model1.coef_[2]
        row["f2_quadratic_coef"] = model2.coef_[2]

    except Exception:
        row["f1_quadratic_coef"] = row["f2_quadratic_coef"] = np.nan

    return row


def compute_spline_coefficients(token_middle, n_points=5):
    """
    compute cubic spline coefficients for f1 and f2

    params
    ------
    token_middle: pd.dataframe
        dataframe containing middle portion of vowel token
    n_points: int
        number of equally spaced points to sample (default 5)

    returns
    -------
    dict
        dictionary containing spline coefficients for f1 and f2
    """
    row = {}

    try:
        x = token_middle["measurement_time"].values
        y1 = token_middle["f1p"].values
        y2 = token_middle["f2p"].values

        # fit cubic splines
        spline1 = UnivariateSpline(x, y1, s=0.5, k=3)  # smoothing factor 0.5, degree 3
        spline2 = UnivariateSpline(x, y2, s=0.5, k=3)  # idem

        # sample equally spaced points
        x_sample = np.linspace(x.min(), x.max(), n_points)

        # extract spline values at sampled points
        for i, xi in enumerate(x_sample):
            row[f"f1_spline_coef_{i + 1}"] = spline1(xi)
            row[f"f2_spline_coef_{i + 1}"] = spline2(xi)

    except Exception:
        # set all spline coefficients to nan if fitting fails
        for i in range(1, n_points + 1):
            row[f"f1_spline_coef_{i}"] = np.nan
            row[f"f2_spline_coef_{i}"] = np.nan

    return row
