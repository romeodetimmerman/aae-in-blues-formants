import os
import pandas as pd
import glob
import shutil
import json


def extract_artist_and_song(filepath):
    """
    extract artist and song names from filepath

    params
    ------
    filepath: str
        path of file

    returns
    -------
    artist: str
        name of artist
    song: str
        title of song
    """
    filename = os.path.basename(filepath)
    base_name, _ = os.path.splitext(filename)
    parts = base_name.split("-so-")
    if len(parts) == 2:
        artist, song = parts
        return artist, song
    else:
        return None, None


def preprocess_files(
    input_vocals_folder="../../data/raw/vocals",
    input_transcriptions_folder="../../data/raw/transcriptions",
    output_vocals_folder="../../data/interim/vocals",
    output_transcriptions_folder="../../data/interim/transcriptions",
    artist_song_key_path="../../data/raw/artist_song_key.json",
    pdf_pattern="*.pdf",
    wav_pattern="*.wav",
    input_csv_path="../../data/raw/perceptive-coding.csv",
    output_csv_path="../../data/interim/perceptive-coding.csv",
    csv_artist_column="artist",
    csv_song_column="song",
):
    """
    preprocess audio and transcription files by assigning unique IDs and organizing them

    params
    ------
    input_vocals_folder: str
        path to folder containing vocal audio files
    input_transcriptions_folder: str
        path to folder containing transcription pdfs
    output_vocals_folder: str
        path to save processed vocal audio files
    output_transcriptions_folder: str
        path to save processed transcription pdfs
    artist_song_key_path: str
        path to the json file containing artist and song id mappings
    pdf_pattern: str
        glob pattern for pdf files
    wav_pattern: str
        glob pattern for wav files
    input_csv_path: str
        path to input csv file with perceptive coding data
    output_csv_path: str
        path to save processed csv file
    csv_artist_column: str
        name of artist column in csv
    csv_song_column: str
        name of song column in csv

    returns
    -------
    none
    """
    # check if input folders exist
    if not os.path.exists(input_vocals_folder):
        print(f"error: input vocals folder does not exist: {input_vocals_folder}")
        return
    if not os.path.exists(input_transcriptions_folder):
        print(
            f"error: input transcriptions folder does not exist: {input_transcriptions_folder}"
        )
        return

    # check if key file exists
    if not os.path.exists(artist_song_key_path):
        print(f"error: artist song key file not found at {artist_song_key_path}")
        return

    # load the artist/song key
    with open(artist_song_key_path, "r") as f:
        id_name_key = json.load(f)

    # create an inverted map: (artist_name, song_name) -> "artist_id-song_id"
    name_to_id_map = {(v["artist"], v["song"]): k for k, v in id_name_key.items()}

    # create output folders if they don't exist
    os.makedirs(output_vocals_folder, exist_ok=True)
    os.makedirs(output_transcriptions_folder, exist_ok=True)

    #####################
    # gather pdf and wav files #
    #####################
    pdf_files = glob.glob(os.path.join(input_transcriptions_folder, pdf_pattern))
    wav_files = glob.glob(os.path.join(input_vocals_folder, wav_pattern))
    all_files = pdf_files + wav_files

    print(f"found {len(pdf_files)} pdf files")
    print(f"found {len(wav_files)} wav files")
    print(f"total files to process: {len(all_files)}")

    if not all_files:
        print(
            "no files found to process. please check your input folders and file patterns."
        )
        return

    #####################################
    # copy the files to <artist_id>-<song_id>.<ext> #
    #####################################
    processed_files_count = 0
    skipped_files_count = 0
    for f in all_files:
        artist, song = extract_artist_and_song(f)
        if artist and song:
            # look up the combined id from the loaded key
            lookup_key = (artist, song)
            if lookup_key in name_to_id_map:
                artist_song_id = name_to_id_map[lookup_key]  # e.g., "001-001"
                _, ext = os.path.splitext(f)
                new_name = f"{artist_song_id}{ext}"

                # determine output folder based on file extension
                if ext.lower() == ".pdf":
                    new_path = os.path.join(output_transcriptions_folder, new_name)
                else:  # wav files
                    new_path = os.path.join(output_vocals_folder, new_name)

                # copy file instead of moving it
                shutil.copy2(f, new_path)
                # print(f"copied: {os.path.basename(f)} -> {new_name}") # optionally keep for verbose output
                processed_files_count += 1
            else:
                print(
                    f"warning: skipping file (artist/song not found in key): {os.path.basename(f)}"
                )
                skipped_files_count += 1
        else:
            print(
                f"warning: skipping file (could not extract artist/song): {os.path.basename(f)}"
            )
            skipped_files_count += 1

    print(
        f"finished processing files. copied: {processed_files_count}, skipped: {skipped_files_count}"
    )

    ##################################
    # process the csv to add the new 3-part ID #
    ##################################

    # check if input csv exists
    if not os.path.exists(input_csv_path):
        print(f"error: input csv file not found at {input_csv_path}")
        return

    # use the full path for the csv file
    df = pd.read_csv(input_csv_path)

    # keep counters for each (artist, song) combination to build the third group (measurement index)
    # key: "artist_id-song_id", value: integer count
    measurement_counters = {}

    # initialize counters based on keys found in the json map
    for artist_song_id in id_name_key.keys():
        measurement_counters[artist_song_id] = 0

    # store final ID
    vowel_ids = []

    for _, row in df.iterrows():
        artist_val = row[csv_artist_column]
        song_val = row[csv_song_column]

        # look up the "artist_id-song_id" string using the inverted map
        lookup_key = (artist_val, song_val)
        if lookup_key in name_to_id_map:
            artist_song_id = name_to_id_map[lookup_key]  # e.g., "001-001"

            # increment measurement counter for this specific "artist_id-song_id"
            measurement_counters.setdefault(
                artist_song_id, 0
            )  # ensure key exists if somehow missed initially
            measurement_counters[artist_song_id] += 1
            measurement_index = f"{measurement_counters[artist_song_id]:03d}"

            # build the final ID string
            vowel_id = f"{artist_song_id}-{measurement_index}"
            vowel_ids.append(vowel_id)

        else:
            # handle case where artist/song from csv is not in the key
            print(
                f"warning: artist '{artist_val}', song '{song_val}' not found in key. assigning default id."
            )
            vowel_ids.append("000-000-000")  # or handle as appropriate

    # add it as a new column to df
    df["vowel_id"] = vowel_ids

    # save the updated csv in the specified output folder
    df.to_csv(output_csv_path, index=False)
    print(f"updated csv saved to {output_csv_path}")


if __name__ == "__main__":
    # assuming the script is run from the project root or paths are adjusted accordingly
    # determine project root based on script location (src/data/preprocess.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # construct absolute paths relative to the project root
    input_vocals = os.path.join(project_root, "data/raw/vocals")
    input_transcriptions = os.path.join(project_root, "data/raw/transcriptions")
    output_vocals = os.path.join(project_root, "data/interim/vocals")
    output_transcriptions = os.path.join(project_root, "data/interim/transcriptions")
    key_path = os.path.join(project_root, "data/raw/artist_song_key.json")
    input_csv = os.path.join(project_root, "data/raw/perceptive_coding.csv")
    output_csv = os.path.join(project_root, "data/interim/perceptive_coding.csv")

    preprocess_files(
        input_vocals_folder=input_vocals,
        input_transcriptions_folder=input_transcriptions,
        output_vocals_folder=output_vocals,
        output_transcriptions_folder=output_transcriptions,
        artist_song_key_path=key_path,
        input_csv_path=input_csv,
        output_csv_path=output_csv,
    )
