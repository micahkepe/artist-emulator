import numpy as np
import sys
import argparse
from keras.models import load_model
from music21 import note, chord, stream, instrument
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

def generate_music(artist, output_path):
    artist = artist.lower()

    # Parameters
    # generate substantial amount of notes
    sequence_length = 100
    output_notes_length = 5000 
    logging.info(f"Generating {output_notes_length} notes for {artist}...")

    # Load the model
    model_path = f'data/{artist}/models/model.h5' 
    model = load_model(model_path)
    logging.info(f"Loaded model from {model_path}")

    # Load the encoding/decoding info
    note_encoder = LabelEncoder()  # Initialize a new encoder
    note_encoder.classes_ = np.load(f'data/{artist}/processed/note_encoder.npy')  # Load the classes
    logging.info(f"Loaded note encoder from data/{artist}/processed/note_encoder.npy")

    # Load a seed sequence
    seed_path = f'data/{artist}/seed/seed.npy'
    seed_sequence = np.load(seed_path)
    logging.info(f"Loaded seed sequence from {seed_path}")

    if len(seed_sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(seed_sequence), 5))
        seed_sequence = np.vstack((padding, seed_sequence))
        logging.info(f"Padded seed sequence to length {sequence_length}")
    else:
        seed_sequence = seed_sequence[-sequence_length:]
        logging.info(f"Truncated seed sequence to length {sequence_length}")

    # Generate music
    output_notes = []

    # generate 500 notes
    for note_index in range(output_notes_length):
        prediction_input = np.reshape(seed_sequence, (1, sequence_length, 5))

        # Predict the attributes of the next note in the sequence
        prediction = model.predict(prediction_input)

        # Get the indices of the attributes from the prediction
        predicted_note_index = np.argmax(prediction[0])
        predicted_note = note_encoder.classes_[predicted_note_index]

        # If the pattern is a chord
        if ('.' in predicted_note) or predicted_note.isdigit():
            notes_in_chord = predicted_note.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            output_notes.append(new_chord)
        # If the pattern is a rest
        elif predicted_note == 'Rest':
            new_note = note.Rest()
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # If the pattern is a note
        else:
            new_note = note.Note(predicted_note)
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Shift the seed sequence
        result_array = np.zeros((1, 5))
        result_array[0, 0] = predicted_note_index  # Store the predicted note
        seed_sequence = np.vstack((seed_sequence[1:], result_array))

    # Create a musical score
    output_score = stream.Score()

    # Add the generated notes to the score
    for note_or_chord in output_notes:
        output_score.append(note_or_chord)
    logging.debug(f"Generated {len(output_notes)} notes")

    # Save the music to a MIDI file
    midi_file_name = f"{output_path}/{artist}_output.mid"
    output_score.write('midi', fp=midi_file_name)
    logging.info(f"Saved generated music to {midi_file_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music using a trained model.')
    parser.add_argument('artist', type=str, help='The name of the artist to generate music for.')
    parser.add_argument('output_path', type=str, help='The output path for the generated MIDI file.')
    args = parser.parse_args()
    generate_music(args.artist, args.output_path)
