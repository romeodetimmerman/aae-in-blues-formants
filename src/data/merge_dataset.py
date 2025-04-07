import pandas as pd
from pathlib import Path


def merge_formant_files():
    """
    merge all formant csv files into a single csv file

    returns
    -------
    None
        saves merged csv to data/interim/merged_formants.csv
    """
    # get project root directory (2 levels up from this script)
    root_dir = Path(__file__).parent.parent.parent

    # get all csv files from raw formants directory
    formant_files = list(root_dir.glob("data/raw/formants/*_FastTrack.csv"))

    if not formant_files:
        raise ValueError("no formant files found in data/raw/formants/")

    # create list to store dataframes
    dfs = []

    # read each csv file and append to list
    for file in formant_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # create interim directory if it doesn't exist
    output_dir = root_dir / "data/interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save merged dataframe
    output_path = output_dir / "merged_formants.csv"
    merged_df.to_csv(output_path, index=False)

    print(f"merged {len(formant_files)} files into {output_path}")


def merge_formants_with_perceptive_coding():
    """
    merge the merged formant csv with the perceptive coding csv
    """
    # get project root directory (2 levels up from this script)
    root_dir = Path(__file__).parent.parent.parent
    interim_dir = root_dir / "data/interim"

    # read the merged formant csv
    formants_df = pd.read_csv(interim_dir / "merged_formants.csv")

    # drop columns
    columns_to_keep = [
        "time",
        "f1",
        "f2",
        "f1p",
        "f2p",
        "vowel",
        "duration",
        "start",
        "end",
    ]
    formants_df = formants_df[columns_to_keep]

    # avoid confusion with categorical time column
    formants_df = formants_df.rename(columns={"time": "measurement_time"})

    # read the perceptive coding csv
    perceptive_df = pd.read_csv(interim_dir / "perceptive_coding.csv")

    # print length of each dataframe
    print(f"formants_df length: {len(formants_df)}")
    print(f"perceptive_df length: {len(perceptive_df)}")

    # merge the two dataframes
    merged_df = pd.merge(
        perceptive_df, formants_df, left_on="vowel_id", right_on="vowel", how="left"
    )

    # check for unmatched rows
    unmatched = merged_df[merged_df["f1"].isna()]
    if len(unmatched) > 0:
        print(
            f"\nwarning: {len(unmatched)} rows in perceptive coding have no matching formant data"
        )
        print("these rows will have NaN values for formant-related columns")
        print("\nfirst few unmatched rows:")
        print(unmatched[["vowel_id"]].head())

        # save unmatched rows to a separate file for inspection
        unmatched_path = interim_dir / "unmatched_rows.csv"
        unmatched.to_csv(unmatched_path, index=False)
        print(f"\nsaved unmatched rows to {unmatched_path}")

    # print length of merged dataframe
    print(f"\nmerged_df length: {len(merged_df)}")
    print(f"matched rows: {len(merged_df) - len(unmatched)}")
    print(f"unmatched rows: {len(unmatched)}")

    # drop rows where f1 or f2 is NaN
    print(f"\nmerged_df length before dropping na rows: {len(merged_df)}")
    merged_df = merged_df.dropna(subset=["f1", "f2"])
    print(f"\nmerged_df length after dropping na rows: {len(merged_df)}")

    # drop the vowel column from the merged dataframe
    merged_df = merged_df.drop(columns=["vowel"])

    # save the merged dataframe
    output_path = interim_dir / "merged_formants_perceptive.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nsaved merged dataframe to {output_path}")


if __name__ == "__main__":
    merge_formant_files()
    merge_formants_with_perceptive_coding()
