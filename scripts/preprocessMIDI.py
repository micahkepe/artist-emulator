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
offsets = []
rests = []

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
            for element in piece.flatten():
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.duration.quarterLength)
                    intensities.append(element.volume.velocity)
                    offsets.append(element.offset)
                    rests.append(0) # default value for no rest
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    durations.append(element.duration.quarterLength)
                    intensities.append(element.volume.velocity)
                    offsets.append(element.offset)
                    rests.append(0) # default value for no rest
                elif isinstance(element, note.Rest):
                    notes.append('Rest') # default value for no note
                    durations.append(element.duration.quarterLength) # duration of rest
                    intensities.append(0) # default value for no note
                    offsets.append(element.offset) # offset of rest
                    rests.append(1) # 1 indicates rest
        except Exception as e:
            logging.error(f"Error occurred while parsing {filename}: {str(e)}")
            continue

# Integer encoding for notes
logging.info("Performing integer encoding...")
note_encoder = LabelEncoder()
notes_encoded = note_encoder.fit_transform(notes)

# Save the encoding
logging.info("Saving the encoding...")
np.save(os.path.join(output_directory, 'note_encoder.npy'), note_encoder.classes_)

# Normalization of durations and intensities
logging.info("Performing normalization...")
scaler = MinMaxScaler()
durations_scaled = scaler.fit_transform(np.array(durations).reshape(-1, 1)).reshape(-1)
intensities_scaled = scaler.fit_transform(np.array(intensities).reshape(-1, 1)).reshape(-1)
offsets_scaled = scaler.fit_transform(np.array(offsets).reshape(-1, 1)).reshape(-1)
rests_scaled = scaler.fit_transform(np.array(rests).reshape(-1, 1)).reshape(-1)

print("Notes:", len(notes_encoded))
print("Durations:", len(durations_scaled))
print("Intensities:", len(intensities_scaled))
print("Offsets:", len(offsets_scaled))
print("Rests:", len(rests_scaled))

# Generate sequences of input and output data
# Each input sequence is a list of 5 elements: note, duration, intensity, offset, rest
logging.info("Generating sequences of input and output data...")
inputs = []
outputs = []
for i in range(0, len(notes_encoded) - sequence_length, 1):
    input_sequence = []
    for j in range(0, sequence_length):
        input_sequence.append([notes_encoded[i + j], durations_scaled[i + j], intensities_scaled[i + j], offsets_scaled[i + j], rests_scaled[i + j]])
    inputs.append(input_sequence)
    outputs.append(notes_encoded[i + sequence_length])

# Split the data into training and testing sets
logging.info("Splitting data into training and testing sets...")

lengths = [len(input) for input in inputs]
if len(set(lengths)) > 1:
    print("Mismatch found in input lengths")
else:
    print("All inputs have the same length")

# Reshape the data for LSTM
inputs = np.array(inputs)
outputs = np.array(outputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 5)) # 5 features

# Split the data
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=test_split, random_state=42)

# Save the preprocessed data
try:
    np.savez(output_directory, inputs_train=inputs_train, outputs_train=outputs_train,
             inputs_test=inputs_test, outputs_test=outputs_test)
    logging.info(f"Preprocessed data saved to: {output_directory}")
except Exception as e:
    logging.error(f"Error occurred while saving preprocessed data: {str(e)}")