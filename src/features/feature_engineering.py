import pandas as pd

from data_loader import load_and_prepare_data, get_vowel_metadata, filter_middle_portion
from distance_features import compute_euclidean_distance_and_deltas
from coefficient_features import (
    compute_linear_slopes,
    compute_quadratic_coefficients,
    compute_spline_coefficients,
)
from derivative_features import compute_derivative_features


def extract_features_for_token(token):
    """
    extract all features for a single vowel token

    params
    ------
    token: pd.dataframe
        dataframe containing measurements for a single vowel token

    returns
    -------
    dict
        dictionary containing all extracted features
    """
    # skip if not enough points
    if len(token) < 10:
        return None

    # get metadata
    row = get_vowel_metadata(token)

    # get middle portion for dynamic analysis
    token_middle, t20, t80 = filter_middle_portion(token)

    # compute dynamic features
    dynamic_features = compute_euclidean_distance_and_deltas(token_middle, t20, t80)
    row.update(dynamic_features)

    # compute linear slopes
    slope_features = compute_linear_slopes(token_middle)
    row.update(slope_features)

    # compute quadratic coefficients
    quadratic_features = compute_quadratic_coefficients(token_middle)
    row.update(quadratic_features)

    # compute spline coefficients
    spline_features = compute_spline_coefficients(token_middle)
    row.update(spline_features)

    # compute derivative features
    derivative_features = compute_derivative_features(token_middle)
    row.update(derivative_features)

    return row


def get_output_columns():
    """
    get list of output columns in correct order

    returns
    -------
    list
        list of column names in desired output order
    """
    return [
        "vowel_id",
        "perceptive_label",
        "vowel_duration",
        "euclid_dist",
        "f1_delta",
        "f2_delta",
        "f1_linear_slope",
        "f2_linear_slope",
        "f1_quadratic_coef",
        "f2_quadratic_coef",
        "f1_spline_coef_1",
        "f1_spline_coef_2",
        "f1_spline_coef_3",
        "f1_spline_coef_4",
        "f1_spline_coef_5",
        "f2_spline_coef_1",
        "f2_spline_coef_2",
        "f2_spline_coef_3",
        "f2_spline_coef_4",
        "f2_spline_coef_5",
        "f1_mean_first_deriv",
        "f1_min_first_deriv",
        "f1_max_first_deriv",
        "f1_std_first_deriv",
        "f1_mean_second_deriv",
        "f1_min_second_deriv",
        "f1_max_second_deriv",
        "f1_std_second_deriv",
        "f2_mean_first_deriv",
        "f2_min_first_deriv",
        "f2_max_first_deriv",
        "f2_std_first_deriv",
        "f2_mean_second_deriv",
        "f2_min_second_deriv",
        "f2_max_second_deriv",
        "f2_std_second_deriv",
    ]


def main():
    """
    main function to orchestrate feature engineering pipeline
    """
    print("loading and preparing data")
    # load and prepare data
    df = load_and_prepare_data("../../data/interim/merged_formants_perceptive.csv")

    print("extracting features for each vowel token")
    # extract features for each vowel token
    features = []
    for vowel_id, token in df.groupby("vowel_id"):
        row = extract_features_for_token(token)
        if row is not None:
            features.append(row)

    print("creating output dataframe")
    # create output dataframe
    df_out = pd.DataFrame(features)
    df_out = df_out[get_output_columns()]

    print("saving results")
    # save results
    df_out.to_csv("../../data/processed/formant_features.csv", index=False)
    print(
        "feature engineering complete, output saved to ../../data/processed/formant_features.csv"
    )


if __name__ == "__main__":
    main()
