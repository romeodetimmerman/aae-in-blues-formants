import pandas as pd
from sklearn.linear_model import LinearRegression

"""
this script classifies PRICE vowels as monophtongs or diphthongs 
based on linear regression slopes for f1 and f2 measurements
"""

####################
# define threshold #
####################

THRESHOLD_F1 = -0.5
THRESHOLD_F2 = 0.5


#################
# linear slopes #
#################


def get_slopes_for_vowel(token):
    """
    fit linear regression models to f1 and f2 measurements and return their respective slopes

    params
    ------
    token: pandas grouped dataframe
           measurements for a particular token

    returns
    -------
    slope_f1, slope_f2: pandas series
                        slopes for f1 and f2 for the given token
    """
    # sort by time (likely redundant, but just to be sure)
    token = token.sort_values("time")

    # convert time to ms
    token["time_ms"] = token["time"] * 1000

    # X is "time" as a 2D array for sklearn
    X = token[["time_ms"]].values

    # fit f1 slope
    y_f1 = token["f1p"].values
    model_f1 = LinearRegression()
    model_f1.fit(X, y_f1)
    slope_f1 = model_f1.coef_[0]

    # fit f2 slope
    y_f2 = token["f2p"].values
    model_f2 = LinearRegression()
    model_f2.fit(X, y_f2)
    slope_f2 = model_f2.coef_[0]

    return pd.Series({"slope_f1": slope_f1, "slope_f2": slope_f2})


def classify_tokens(row, threshold_f1=THRESHOLD_F1, threshold_f2=THRESHOLD_F2):
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


##############
# run script #
##############


def main():
    # read data
    df = pd.read_csv("../../data/raw/EN_03.csv")
    df = df[df["vowel"] == "ai"]

    # group by each vowel token and calculate slopes
    df = df.groupby(["token"], as_index=False).apply(
        get_slopes_for_vowel, include_groups=False
    )

    # classify vowels based on slopes
    df["classification"] = df.apply(classify_tokens, axis=1)

    # print results
    print(df)
    print()
    print(df["classification"].value_counts())


if __name__ == "__main__":
    main()
