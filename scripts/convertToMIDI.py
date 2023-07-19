import os
import sys
import logging
from basic_pitch.inference import predict_and_save

# Set up the logger to info level to display informative messages
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Assign input and output directories
input_directory = sys.argv[1]
output_directory = sys.argv[2]

# Create a directory to store the converted MIDI files if it doesn't already exist
os.makedirs(output_directory, exist_ok=True)

# Store converted filenames in a set for faster lookups
converted_files = set()

# Scan through all the MIDI files in the output directory
for filename in os.listdir(output_directory):
    if filename.endswith("_basic_pitch.mid"):
        # Save the filename without the extension for future comparison
        converted_files.add(filename[:-16])

# Now, go through all the mp3 files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".mp3"):
        # Check if the mp3 file has already been converted by comparing filenames
        if filename[:-4] in converted_files:
            logging.info("Skipping " + filename + " because it has already been converted")
            continue

        input_audio_path = os.path.join(input_directory, filename)
        # Transcribe the mp3 to a MIDI file using basic_pitch
        predict_and_save(
            [input_audio_path],
            output_directory,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
        )
        logging.info("Converted " + filename)
