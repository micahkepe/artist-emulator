import numpy as np
import sys
import argparse
from keras.models import load_model
from music21 import note, chord, stream, instrument
from sklearn.preprocessing import LabelEncoder

def generate_music(artist, output_path):
    # Parameters
    sequence_length = 100
    output_notes_length = 500

    # Load the model
    model_path = f'data/{artist}/models/model.h5' 
    model = load_model(model_path)

    # Load the encoding/decoding info
    note_encoder = LabelEncoder()  # Initialize a new encoder
    note_encoder.classes_ = np.load('note_classes.npy')  # Load the classes

    # Load a seed sequence
    seed_path = 'seed.npz'  # Adjust this as necessary
    seed_data = np.load(seed_path)
    seed_sequence = seed_data['inputs_train'][0]

    # Generate music
    output_notes = []

    # generate 500 notes
    for note_index in range(output_notes_length):
        prediction_input = np.reshape(seed_sequence, (1, sequence_length, 5))

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = note_encoder.inverse_transform([index])
        output_notes.append(result)

        # Shift the seed sequence
        seed_sequence = np.vstack((seed_sequence[1:], result))

    # Convert the output from tokens to note/chord objects
    output_score = stream.Score()
    for pattern in output_notes:
        # pattern is a chord
        if ('.' in pattern[0]) or pattern[0].isdigit():
            notes_in_chord = pattern[0].split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            output_score.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern[0])
            new_note.storedInstrument = instrument.Piano()
            output_score.append(new_note)

    # Save the music to a MIDI file
    midi_file_name = f"{output_path}/{artist}_output.mid"
    output_score.write('midi', fp=midi_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music using a trained model.')
    parser.add_argument('artist', type=str, help='The name of the artist to generate music for.')
    parser.add_argument('output_path', type=str, help='The output path for the generated MIDI file.')
    args = parser.parse_args()
    generate_music(args.artist, args.output_path)
