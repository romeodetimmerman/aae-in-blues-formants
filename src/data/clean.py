import os
import pandas as pd
import glob

"""
this script prepares the formant data to be merged with the perceptive coding data
"""

#########
# clean #
#########


def rename_files_and_update_csv(
    data_folder="../../data/interim/",
    pdf_pattern="*.pdf",
    wav_pattern="*.wav",
    csv_name="data.csv",
    output_csv_name="data_updated.csv",
    csv_artist_column="artist",
    csv_song_column="song",
):
    """
    (i) renames all PDF/WAV files in 'data_folder' based on <artist>-so-<song>.<ext>.
    (ii) creates a 3-part ID (artist_id-song_id-token_id), where:
       - artist_id = 3-digit ID for each unique artist
       - song_id   = 3-digit ID for each song (3 songs per artist)
       - token_id  = 3-digit counter for each row in csv referencing (artist, song).
    (iii) writes the updated csv (with the new ID column) to 'output_csv_name' in the same folder.

    params
    ------
    data_folder: str
                 path to the folder containing files
    pdf_pattern: str
                 file naming convention for pdf files
    wav_pattern: str
                 file naming convention for wav files
    csv_name: str
              name of input csv file
    output_csv_name: str
                     name of output csv file
    csv_artist_column: str
                       name of column containing artist name
    csv_song_column: str
                     name of column containing song title

    returns
    -------
    none
    """

    ############################
    # gather pdf and wav files #
    ############################
    pdf_files = glob.glob(os.path.join(data_folder, pdf_pattern))
    wav_files = glob.glob(os.path.join(data_folder, wav_pattern))
    all_files = pdf_files + wav_files

    ############################################################################
    # build dictionaries to map artist -> artist_id, (artist, song) -> song_id #
    ############################################################################
    artist_id_map = {}
    song_id_map_per_artist = {}

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

    ###################################################
    # rename the files to <artist_id>-<song_id>.<ext> #
    ###################################################
    for f in all_files:
        artist, song = extract_artist_and_song(f)
        if artist and song:
            artist_id = artist_id_map[artist]
            song_id = song_id_map_per_artist[artist][song]
            _, ext = os.path.splitext(f)
            new_name = f"{artist_id}-{song_id}{ext}"
            new_path = os.path.join(data_folder, new_name)
            # rename in-place
            os.rename(f, new_path)
            print(f"renamed: {os.path.basename(f)} -> {new_name}")
        else:
            print(f"skipping file (no matching pattern): {os.path.basename(f)}")

    ############################################
    # process the csv to add the new 3-part ID #
    ############################################
    csv_path = os.path.join(data_folder, csv_name)
    df = pd.read_csv(csv_path)

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
    df["ID"] = unique_ids

    # save the updated csv
    output_csv_path = os.path.join(data_folder, output_csv_name)
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")


##############
# run script #
##############


def main():
    rename_files_and_update_csv(
        data_folder="../../data/interim/lyrics",
        pdf_pattern="*.pdf",
        wav_pattern="*.wav",
        csv_name="rdt-data-original.csv",
        output_csv_name="rdt-data-processed.csv",
        csv_artist_column="artist",
        csv_song_column="song",
    )


if __name__ == "__main__":
    main()
