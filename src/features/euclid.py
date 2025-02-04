import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
this script classifies PRICE vowels as monophtongs or diphthongs 
based on the euclidean distance in the f1-f2 space
"""

####################
# define threshold #
####################

THRESHOLD = 250

######################
# euclidean distance #
######################


def compute_distance(token):
    """
    compute f1 and f2 at 20% and 80% mark of vowel
    and calculate euclidean distance between these two points

    params
    ------
    token: pandas grouped dataframe
           measurements for a particular token


    returns
    -------
    distance: pandas series
              euclidean distance between 20% and 80% mark of vowel

    """
    duration = token["duration"].iloc[0]
    t20 = 0.2 * duration
    t80 = 0.8 * duration

    # indices of rows whose 'time' is closest to 20% and 80%
    idx_20 = (token["time"] - t20).abs().idxmin()
    idx_80 = (token["time"] - t80).abs().idxmin()

    # extract F1, F2 at 20% and 80%
    f1_20, f2_20 = token.loc[idx_20, "f1p"], token.loc[idx_20, "f2p"]
    f1_80, f2_80 = token.loc[idx_80, "f1p"], token.loc[idx_80, "f2p"]

    # euclidean distance between (f1_20, f2_20) and (f1_80, f2_80)
    distance = np.sqrt((f1_20 - f1_80) ** 2 + (f2_20 - f2_80) ** 2)

    # return a series (so pandas can merge results across tokens)
    return pd.Series({"distance": distance})


def classify_vowels(row, threshold=THRESHOLD):
    """
    classifify vowels based on euclidean distance

    params
    ------
    row: pandas series
    threshold: int
               global variable specifying cutoff for euclidean distance

    returns
    -------
    classification: bool
                    monophthong or diphthong
    """

    if row["distance"] > threshold:
        return "diphthong"
    else:
        return "monophthong"


##############
# run script #
##############


def main():
    # read data
    df = pd.read_csv("../../data/raw/EN_06.csv")

    df = df[df["vowel"] == "aa"]

    # apply euclidean distance function to each group
    df = df.groupby(["token"], as_index=False).apply(
        compute_distance, include_groups=False
    )

    # apply threshold to classify vowels
    df["classification"] = df.apply(classify_vowels, axis=1)

    # print results
    print(df)
    print()

    print(df["distance"].describe())
    print()

    sns.histplot(df["distance"], bins=20)
    plt.show()

    print(df["classification"].value_counts())
    plt.show()

    print(df[df["distance"] > 350])


if __name__ == "__main__":
    main()
