import os
import subprocess
import shutil


def isolate_vocals(input_folder, output_folder):
    """
    isolate vocals from mp3 files in the input folder

    args:
        input_folder (str): path to folder containing mp3 files
        output_folder (str): path to folder where wav files will be saved

    returns:
        none: saves isolated vocals as wav files in output folder
    """
    # ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # get list of mp3 files in the input folder
    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("no mp3 files found in the input folder.")
        return

    for mp3_file in mp3_files:
        input_path = os.path.join(input_folder, mp3_file)
        temp_output_folder = os.path.join(output_folder, "htdemucs")
        print(f"processing {mp3_file}...")

        try:
            # run demucs for vocal isolation
            subprocess.run(
                [
                    "demucs",
                    "--two-stems",
                    "vocals",
                    "-o",
                    output_folder,
                    input_path,
                ],
                check=True,
            )

            # locate and move vocals.wav to final output folder
            temp_subfolder = os.path.join(
                temp_output_folder, os.path.splitext(mp3_file)[0]
            )
            vocals_file = os.path.join(temp_subfolder, "vocals.wav")
            final_vocals_path = os.path.join(
                output_folder, os.path.splitext(mp3_file)[0] + ".wav"
            )

            if os.path.exists(vocals_file):
                shutil.move(vocals_file, final_vocals_path)
                print(f"saved vocals for {mp3_file} as {final_vocals_path}")
            else:
                print(f"error: vocals.wav not found for {mp3_file}")

        except subprocess.CalledProcessError as e:
            print(f"error processing {mp3_file}: {e}")

        finally:
            # clean up temporary htdemucs folder
            shutil.rmtree(temp_output_folder, ignore_errors=True)


##########
# run script #
##########


def main():
    input_folder = "../../data/raw/lyrics"
    output_folder = "../../data/interim/vocals"
    isolate_vocals(input_folder, output_folder)


if __name__ == "__main__":
    main()
