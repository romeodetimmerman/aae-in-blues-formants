import pandas as pd
from utils import normalize_speaker_formants


def load_and_prepare_data(filepath):
    """
    load and prepare formant data for feature engineering

    params
    ------
    filepath: str
        path to the merged formants csv file

    returns
    -------
    pd.dataframe
        prepared dataframe with normalized formants
    """
    # load data
    df = pd.read_csv(filepath)

    # select relevant columns
    keep_cols = [
        "vowel_id",
        "artist",
        "measurement_time",
        "f1p",
        "f2p",
        "duration",
        "aae_realization",
    ]
    df = df[keep_cols].copy()

    # normalize formants per speaker
    df = normalize_speaker_formants(
        df, speaker_col="artist", formant_cols=["f1p", "f2p"]
    )

    return df


def get_vowel_metadata(token):
    """
    extract metadata for a vowel token

    params
    ------
    token: pd.dataframe
        dataframe containing measurements for a single vowel token

    returns
    -------
    dict
        metadata dictionary with vowel_id, perceptive_label, and duration
    """
    row = {}
    row["vowel_id"] = token["vowel_id"].iloc[0]

    # map numeric aae_realization to string labels
    aae_val = token["aae_realization"].iloc[0]
    if aae_val == 0:
        row["perceptive_label"] = "diphthong"
    elif aae_val == 1:
        row["perceptive_label"] = "monophthong"
    else:
        row["perceptive_label"] = "unknown"

    row["vowel_duration"] = token["duration"].iloc[0]

    return row


def filter_middle_portion(token):
    """
    filter token to middle portion for dynamic analysis

    params
    ------
    token: pd.dataframe
        dataframe containing measurements for a single vowel token

    returns
    -------
    pd.dataframe
        filtered dataframe containing only middle portion
    """
    # get 20% and 80% time points
    t20 = token["measurement_time"].min() + 0.2 * (
        token["measurement_time"].max() - token["measurement_time"].min()
    )
    t80 = token["measurement_time"].min() + 0.8 * (
        token["measurement_time"].max() - token["measurement_time"].min()
    )

    # filter to middle portion
    token_middle = token[
        (token["measurement_time"] >= t20) & (token["measurement_time"] <= t80)
    ]

    return token_middle, t20, t80
