import numpy as np
import sys
import logging
from music21 import instrument, note, chord, stream
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import datetime

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()
timestamp = sys.argv[2]  # fetch the timestamp from command line arguments

# Load the seed
seed = np.load(f'data/{artist}/seed/seed_{timestamp}.npy')
logging.info(f"Loaded seed for {artist}.")

# Load the note encoder
note_encoder = LabelEncoder()
note_encoder.classes_ = np.load(f'data/{artist}/preprocessed/note_encoder_1689734573.npy') # change this to the latest processed data file
logging.info(f"Loaded note encoder for {artist}.")

# map the seed notes to discrete integers for inverse_transform
discrete_output = np.round(seed[0, :, 0]).astype(int)

# Get labels from the integer encoding of the seed notes
logging.info("Getting labels from the integer encoding of the seed notes...")
seed_notes = note_encoder.inverse_transform(discrete_output)
logging.debug('Seed notes:', seed_notes)

# Create a steam object for the seed
midi_stream = stream.Stream()

# Add the notes and chords to the stream object
logging.info("Adding notes and chords to the stream object...")
for element in seed_notes:
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

# Write the stream object to a MIDI file
logging.info("Writing the stream object to a MIDI file...")
midi_stream.write('midi', fp=f'data/{artist}/seed/seed_{timestamp}.mid')
