import pandas as pd
import euclid_distances
import linear_slopes
import formant_deltas
from utils import seconds_to_milliseconds, train_test


def process_data(df_measurements):
    """
    aggregate per-vowel features (distance, slopes, deltas) and split into train/test.

    calculates euclidean distance, linear slopes, and relative formant deltas
    for each vowel based on its formant measurements over time. creates a
    dataframe with one row per vowel containing these features and the vowel's
    label. splits this aggregated dataframe into training and testing sets.

    params
    ------
    df_measurements:
        dataframe containing formant measurements per vowel token over time.
        expects columns like 'measurement_time', 'duration', 'vowel_id',
        'f1p', 'f2p', 'aae_realization'.

    returns
    -------
    two dataframes:
        - df_vowel_train: aggregated per-vowel features for training set.
        - df_vowel_test: aggregated per-vowel features for test set.
    """
    # convert time columns to milliseconds
    df_measurements["time_ms"] = seconds_to_milliseconds(
        df_measurements, seconds_column="measurement_time"
    )
    df_measurements["duration_ms"] = seconds_to_milliseconds(
        df_measurements, seconds_column="duration"
    )

    # compute euclidean distance per vowel token
    euclid_df = df_measurements.groupby("vowel_id", as_index=False).apply(
        euclid_distances.compute_distance,
        f1_column="f1p",
        f2_column="f2p",
        time_column="time_ms",
        duration_column="duration_ms",
        include_groups=False,
    )

    # compute linear slopes per vowel token
    slopes_df = df_measurements.groupby("vowel_id", as_index=False).apply(
        linear_slopes.compute_slopes,
        f1_column="f1p",
        f2_column="f2p",
        time_column="time_ms",
        duration_column="duration_ms",
        include_groups=False,
    )

    # compute relative deltas per vowel token
    deltas_df = df_measurements.groupby("vowel_id", as_index=False).apply(
        formant_deltas.compute_deltas,
        f1_column="f1p",
        f2_column="f2p",
        time_column="time_ms",
        duration_column="duration_ms",
        include_groups=False,
    )

    # create base per-vowel dataframe by merging aggregated features
    df_vowel = euclid_df.merge(slopes_df, on="vowel_id", how="inner")
    df_vowel = df_vowel.merge(deltas_df, on="vowel_id", how="inner")

    # get unique labels per vowel_id from the original measurements dataframe
    # assuming label ('aae_realization') is constant for all measurements of a single vowel_id
    vowel_labels = df_measurements[["vowel_id", "aae_realization"]].drop_duplicates()

    # merge labels into the per-vowel dataframe
    df_vowel = pd.merge(
        df_vowel, vowel_labels, on="vowel_id", how="left"
    )  # use left merge to keep all aggregated rows

    # handle potential missing labels if any vowel_id didn't have a label
    if df_vowel["aae_realization"].isnull().any():
        print("warning: missing labels found for some vowel_ids after aggregation.")
        # decide how to handle missing labels, e.g., drop rows or impute
        df_vowel.dropna(subset=["aae_realization"], inplace=True)

    # split the aggregated per-vowel data into training and test sets
    df_vowel_train, df_vowel_test = train_test(df_vowel, label_col="aae_realization")

    return df_vowel_train, df_vowel_test


if __name__ == "__main__":
    # load the raw measurement data
    df_measurements = pd.read_csv("../../data/interim/merged_formants_perceptive.csv")

    # process data: aggregate features per vowel and split into train/test
    df_train, df_test = process_data(df_measurements)

    # save the aggregated train and test sets
    output_dir = "../../data/processed"
    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_data.csv", index=False)

    print(
        f"aggregated data processing completed, saved train/test sets to {output_dir}"
    )
