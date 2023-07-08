import os
import sys
import logging
from basic_pitch.inference import predict_and_save

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# directories
input_directory = sys.argv[1]
output_directory = sys.argv[2]

# Set to store converted filenames
converted_files = set()

# Iterate through all the MIDI files in the output directory
for filename in os.listdir(output_directory):
    if filename.endswith("_basic_pitch.mid"):
        converted_files.add(filename[:-16])  # Remove "_basic_pitch.mid" extension

# iterate through all the mp3 files
for filename in os.listdir(input_directory):
    if filename.endswith(".mp3"):
        # Skip if the file has already been converted
        if filename[:-4] in converted_files:
            logging.info("Skipping " + filename + " because it has already been converted")
            continue

        input_audio_path = os.path.join(input_directory, filename)
        # basic-pitch transcribes the mp3 to a MIDI file
        predict_and_save(
            [input_audio_path],
            output_directory,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
        )
        logging.info("Converted " + filename)
