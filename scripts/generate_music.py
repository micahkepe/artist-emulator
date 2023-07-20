import numpy as np
import os
import sys
import random
import logging
from keras.models import load_model
from music21 import converter, instrument, note, chord, stream, tempo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import datetime

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()
timestamp = sys.argv[2]  # fetch the timestamp for the model from command line arguments

# Load the seed and the trained model
seed = np.load(f'data/{artist}/seed/seed_20230720090243.npy') # change this to the latest seed file
model = load_model(f'data/{artist}/models/model_{timestamp}.h5')
logging.info(f"Loaded seed and model for {artist}.")

# Load the preprocessed data
data = np.load(f'data/{artist}/preprocessed/preprocessed_data_1689734573.npz', allow_pickle=True) # change this to the latest preprocessed data file

# Load the note encoder
note_encoder = LabelEncoder()
note_encoder.classes_ = np.load(f'data/{artist}/preprocessed/note_encoder_1689734573.npy') # change this to the latest processed data file
logging.info(f"Loaded preprocessed data and note encoder for {artist}.")

# Start the generated music with the seed
generated_music = seed.flatten().tolist()

# Generate 500 notes
logging.info("Generating notes...")
for i in range(500):
    prediction = model.predict(seed, verbose=0)

    index = np.random.choice(range(len(note_encoder.classes_)), p=prediction[0])
    generated_music.append(index)

    # Create a new array with the same shape as the other features
    new_feature = np.zeros_like(seed[0][0])
    new_feature[0] = index  # set the first feature as the predicted note index

    # Append the new feature to the seed sequence and discard the first element
    seed = np.append(seed[0][1:], [new_feature], axis=0)

    # Reshape the sequence
    seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))

# map the generated notes to discrete integers for inverse_transform
discrete_output = np.round(generated_music).astype(int)

# Get labels from the integer encoding of the generated notes
logging.info("Getting labels from the integer encoding of the generated notes...")
generated_notes = note_encoder.inverse_transform(discrete_output)
print('Generated notes:', generated_notes)

# Create a steam object for the generated music
midi_stream = stream.Stream()

# Add the notes and chords to the stream object
logging.info("Adding notes and chords to the stream object...")
for element in generated_notes:
    if element == "":
        logging.warning("Skipping blank note name.")
        continue
    elif '.' in element: # if the element is a chord
        notes_in_chord = element.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            if current_note.isdigit():
                new_note = note.Note(int(current_note))
            else:
                new_note = note.Note(current_note)  # directly use pitch notation if it's not a digit
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        midi_stream.append(new_chord)
    elif element == "Rest":
        new_note = note.Rest()
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)
    else: # if the element is a note
        if element.isdigit():
            new_note = note.Note(int(element))
        else:
            new_note = note.Note(element)  # directly use pitch notation if it's not a digit
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)


# Set the tempo of the generated music to 130 bpm
midi_stream.append(tempo.MetronomeMark(number=130))

# Write the stream object to a MIDI file
logging.info("Writing the stream object to a MIDI file...")
midi_stream.write('midi', fp=f'data/{artist}/output/generated_music_{timestamp}.mid')

