import numpy as np
import os
import sys
import random
import logging
from keras.models import load_model
from music21 import converter, instrument, note, chord, stream, tempo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the seed and the trained model
seed = np.load(f'data/{artist}/seed/seed.npy')
model = load_model(f'data/{artist}/models/model.h5')
logging.info(f"Loaded seed and model for {artist}.")

# Load the preprocessed data
data = np.load(f'data/{artist}/processed/processed.npz', allow_pickle=True)

# Load the note encoder
note_encoder = LabelEncoder()
note_encoder.classes_ = np.load(f'data/{artist}/processed/note_encoder.npy')
logging.info(f"Loaded preprocessed data and note encoder for {artist}.")

# Start the generated music with the seed
generated_music = seed.flatten().tolist()

# Generate 500 notes
logging.info("Generating notes...")
for i in range(500):
    prediction = model.predict(seed, verbose=0)

    index = np.argmax(prediction)
    generated_music.append(index)

    # Create a new array with the same shape as the other features
    new_feature = np.zeros_like(seed[0][0])
    new_feature[0] = index  # set the first feature as the predicted note index

    # Append the new feature to the seed sequence and discard the first element
    seed = np.append(seed[0][1:], [new_feature], axis=0)

    # Reshape the sequence
    seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))


# Get labels from the integer encoding of the generated notes
logging.info("Getting labels from the integer encoding of the generated notes...")
known_labels = note_encoder.classes_


# Reverse the integer encoding of the generated notes to get notes and chords  
generated_notes = [note_encoder.inverse_transform([x]) if x in known_labels else 'C' for x in generated_music]

# Create a steam object for the generated music
midi_stream = stream.Stream()

# Add the notes and chords to the stream object
logging.info("Adding notes and chords to the stream object...")
for element in generated_notes:
    if '.' in generated_notes: # if the element is a chord
        notes_in_chord = element.split('.')
        chord_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        midi_stream.append(new_chord)
    elif element == "Rest":
        new_note = note.Rest()
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)
    else: # if the element is a note
        new_note = note.Note(element)
        new_note.storedInstrument = instrument.Piano()
        midi_stream.append(new_note)

# Set the tempo of the generated music to 130 bpm
midi_stream.append(tempo.MetronomeMark(number=130))

# Write the stream object to a MIDI file
logging.info("Writing the stream object to a MIDI file...")
midi_stream.write('midi', fp=f'data/{artist}/output/generated_music.mid')

