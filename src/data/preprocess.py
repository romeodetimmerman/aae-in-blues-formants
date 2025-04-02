import os
import pandas as pd
import glob
import shutil


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

    ######################################################
    # build dictionaries to map artist -> artist_id, (artist, song) -> song_id #
    ######################################################
    artist_id_map = {}
    song_id_map_per_artist = {}

    # keep a list of unique (artist, song) tuples while parsing files
    artist_song_pairs = []

    for f in all_files:
        artist, song = extract_artist_and_song(f)
        if artist and song:
            artist_song_pairs.append((artist, song))

    # get unique pairs
    artist_song_pairs = list(set(artist_song_pairs))

    # assign a unique 3-digit ID to each artist
    # for each artist, assign a unique 3-digit ID to each of their songs
    current_artist_id = 1
    for artist, song in artist_song_pairs:
        # create an ID for the artist if not already assigned
        if artist not in artist_id_map:
            artist_id_map[artist] = f"{current_artist_id:03d}"
            song_id_map_per_artist[artist] = {}
            current_artist_id += 1

    # assign song IDs per artist, assuming each artist has exactly 3 songs
    for artist, song in artist_song_pairs:
        # if not assigned a song ID yet, assign the next 3-digit
        if song not in song_id_map_per_artist[artist]:
            # the number of songs already assigned to this artist
            current_song_count = len(song_id_map_per_artist[artist])
            song_id_map_per_artist[artist][song] = f"{(current_song_count + 1):03d}"

    #######################################
    # copy the files to <artist_id>-<song_id>.<ext> #
    #######################################
    for f in all_files:
        artist, song = extract_artist_and_song(f)
        if artist and song:
            artist_id = artist_id_map[artist]
            song_id = song_id_map_per_artist[artist][song]
            _, ext = os.path.splitext(f)
            new_name = f"{artist_id}-{song_id}{ext}"

            # determine output folder based on file extension
            if ext.lower() == ".pdf":
                new_path = os.path.join(output_transcriptions_folder, new_name)
            else:  # wav files
                new_path = os.path.join(output_vocals_folder, new_name)

            # copy file instead of moving it
            shutil.copy2(f, new_path)
            print(f"copied: {os.path.basename(f)} -> {new_name}")
        else:
            print(f"skipping file (no matching pattern): {os.path.basename(f)}")

    ##################################
    # process the csv to add the new 3-part ID #
    ##################################

    # use the full path for the csv file
    df = pd.read_csv(input_csv_path)

    # keep counters for each (artist, song) combination to build the third group (token)
    # key: (artist, song), value: integer count
    token_counters = {}

    # initialize counters to 0 for all known (artist, song) combos discovered from the files
    # handle combos not seen in csv file separately
    for artist, song in artist_song_pairs:
        token_counters[(artist, song)] = 0

    # store final ID
    unique_ids = []

    for _, row in df.iterrows():
        artist_val = row[csv_artist_column]
        song_val = row[csv_song_column]

        # ensure there is an ID for this artist
        if artist_val in artist_id_map:
            artist_id = artist_id_map[artist_val]
        else:
            # handle missing or unknown artist
            artist_id = "000"

        # ensure there is an ID for this song under the given artist
        if (
            artist_val in song_id_map_per_artist
            and song_val in song_id_map_per_artist[artist_val]
        ):
            song_id = song_id_map_per_artist[artist_val][song_val]
        else:
            # handle missing or unknown song
            song_id = "000"

        # increment token counter for this specific (artist, song)
        token_counters.setdefault(
            (artist_val, song_val), 0
        )  # if not present, default 0
        token_counters[(artist_val, song_val)] += 1
        token_id = f"{token_counters[(artist_val, song_val)]:03d}"

        # build the final ID string
        unique_id = f"{artist_id}-{song_id}-{token_id}"
        unique_ids.append(unique_id)

    # add it as a new column to df
    df["vowel_id"] = unique_ids

    # save the updated csv in the specified output folder
    df.to_csv(output_csv_path, index=False)
    print(f"updated csv saved to {output_csv_path}")


if __name__ == "__main__":
    preprocess_files()
