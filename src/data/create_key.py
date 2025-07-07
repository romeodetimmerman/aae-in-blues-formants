import pandas as pd
import json
import os


def create_id_name_key(input_csv_path: str, output_json_path: str):
    """
    extracts artist/song IDs and names from csv and saves as json key

    params
    ------
    input_csv_path: str
        path to the input csv file
    output_json_path: str
        path to save the output json file

    returns
    -------
    none: saves the key to a json file
    """
    # read the csv file
    df = pd.read_csv(input_csv_path)

    # initialize an empty dictionary for the key
    id_name_key = {}

    # iterate over unique combinations of vowel_id, artist, and song
    for _, row in df[["vowel_id", "artist", "song"]].drop_duplicates().iterrows():
        vowel_id = row["vowel_id"]
        artist_name = row["artist"]
        song_name = row["song"]

        # extract artist and song ids from vowel_id
        parts = vowel_id.split("-")
        if len(parts) >= 2:
            artist_id = parts[0]
            song_id = parts[1]

            # store the mapping, converting tuple keys to string for json compatibility
            key = f"{artist_id}-{song_id}"
            id_name_key[key] = {"artist": artist_name, "song": song_name}

    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # save the key to a json file
    with open(output_json_path, "w") as f:
        json.dump(id_name_key, f, indent=4)

    print(f"key saved to {output_json_path}")


if __name__ == "__main__":
    # define input and output paths relative to project root
    INPUT_CSV = "data/interim/merged_formants_perceptive.csv"
    OUTPUT_JSON = "data/raw/artist_song_key.json"

    # get the absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(
        os.path.dirname(script_dir)
    )  # assumes script is in src/data
    abs_input_csv = os.path.join(project_root, INPUT_CSV)
    abs_output_json = os.path.join(project_root, OUTPUT_JSON)

    # check if input file exists
    if not os.path.exists(abs_input_csv):
        print(f"input file not found at {abs_input_csv}")
    else:
        create_id_name_key(abs_input_csv, abs_output_json)
