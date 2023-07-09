import os
import sys
import logging
import numpy as np
from music21 import converter, note, chord
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Directories
input_directory = sys.argv[1]
output_directory = sys.argv[2]

# Parameters for preprocessing
sequence_length = 100  # Length of input sequences
test_split = 0.2  # Percentage of data to use for testing

# Initialize lists for data
notes = []
durations = []
intensities = []

# Iterate through all the MIDI files in the input directory
logging.info(f"Preprocessing MIDI files from: {input_directory}")
for filename in os.listdir(input_directory):
    if filename.endswith(".mid"):
        midi_path = os.path.join(input_directory, filename)
        
       # Parsing the MIDI file
        try:
            piece = converter.parse(midi_path)
            logging.info(f"Parsing {filename}...")

            # Extracting notes and attributes
            for element in piece.flat:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.duration.quarterLength)
                    intensities.append(element.volume.velocity)
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    durations.append(element.duration.quarterLength)
                    intensities.append(element.volume.velocity)
        except Exception as e:
            logging.error(f"Error occurred while parsing {filename}: {str(e)}")
            continue

# Integer encoding for notes
logging.info("Performing integer encoding...")
note_encoder = LabelEncoder()
notes_encoded = note_encoder.fit_transform(notes)

# Normalization of durations and intensities
logging.info("Performing normalization...")
scaler = MinMaxScaler()
durations_scaled = scaler.fit_transform(np.array(durations).reshape(-1, 1))
intensities_scaled = scaler.fit_transform(np.array(intensities).reshape(-1, 1))

# Generate sequences of input and output data
inputs = []
outputs = []

logging.info("Generating sequences of input and output data...")
for i in range(len(notes_encoded) - sequence_length):
    inputs.append(notes_encoded[i:i + sequence_length])
    outputs.append(notes_encoded[i + sequence_length])

# Split the data into training and testing sets
logging.info("Splitting data into training and testing sets...")
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=test_split, random_state=42)

# Save the preprocessed data
try:
    np.savez(output_directory, inputs_train=inputs_train, outputs_train=outputs_train,
             inputs_test=inputs_test, outputs_test=outputs_test)
    logging.info(f"Preprocessed data saved to: {output_directory}")
except Exception as e:
    logging.error(f"Error occurred while saving preprocessed data: {str(e)}")