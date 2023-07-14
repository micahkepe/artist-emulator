import numpy as np
import sys
import argparse
from keras.models import load_model
from music21 import note, chord, stream, instrument
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the preprocessed data and the model
logging.info(f"Loading model and data for {artist}...")
data = np.load(f'data/{artist}/preprocessed_data.npz', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = data['classes']
model = load_model(f'data/{artist}/models/model.h5')

# Load the seed sequence
logging.info("Loading seed sequence...")
seed = np.load(f'data/{artist}/seed/seed.npy')
sequence = list(seed)

# Generate music
logging.info("Generating music...")
for i in range(100):
    prediction = model.predict(np.array([sequence[-100:]]))[0]
    index = np.argmax(prediction)
    sequence.append(label_encoder.transform([index])[0])

# Create a music21 stream and add notes/chords to it
output_notes = []
logging.info("Creating music stream...")
for element in sequence:
    element = label_encoder.inverse_transform([element])[0]
    # Check whether the element is a note or a chord
    if ('.' in element) or element.isdigit():
        output_notes.append(note.Note(element))
    else:
        output_notes.append(chord.Chord(element))

# Create a music21 stream object from the generated notes
midi_stream = stream.Stream(output_notes)

# Save to MIDI file
logging.info("Saving generated music to MIDI file...")
midi_stream.write('midi', fp=f'data/{artist}/output/output.mid')

print(f"Generated music saved to data/{artist}/output/output.mid.")
