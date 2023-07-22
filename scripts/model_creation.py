import glob
import sys 
import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.utils import to_categorical as np_utils
from keras.callbacks import ModelCheckpoint
import logging


def get_notes():
    """ 
    Get all the notes and chords from the midi files in the ./data directory

    Returns:
        notes: List of notes and chords found in the training midi files
    """
    notes = []
    note_file_path = f'data/{artist}/notes/notes_file.pkl' # change this to the latest notes file
    
    # If the notes file exists, load it
    if os.path.exists(note_file_path):
        logging.info(f"Loading notes from {note_file_path}...")
        with open(note_file_path, 'rb') as filepath:
            notes = pickle.load(filepath)
        logging.info(f"Notes loaded from {note_file_path}")
        return notes
    
    # If the notes file doesn't exist, create it
    for file in glob.glob(f"data/{artist}/rawMIDI/*.mid"):

        # Load the midi file
        midi = converter.parse(file)
        logging.info(f"Parsing {file}...")

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        # Loop through all the notes and chords and add them to the notes list
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append('Rest')

    # Save the notes list to a file, creating the data directory if it doesn't exist
    os.makedirs(os.path.dirname(f'data/{artist}/notes/'), exist_ok=True)

    with open(note_file_path, 'wb') as filepath:
        pickle.dump(notes, filepath)
    logging.info(f"Notes saved to {note_file_path}")

    return notes


def prepare_sequences(notes, n_vocab):
    """ 
    Prepare the sequences used by the Neural Network
    
    Args:
        notes: List of notes and chords
        n_vocab: Number of unique notes and chords in the notes list

    Returns:
        network_input: Input sequences for the neural network
        network_output: Output sequences for the neural network
    """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # get amount of pitch names
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    # one hot encode the output vectors
    network_output = np_utils(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ 
    Create the structure of the neural network
    
    Args:
        network_input: Input sequences for the neural network
        n_vocab: Number of unique notes and chords in the notes list
        
    Returns:
        model: Sequential model built with Keras
    """
    # Create the structure of the neural network
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    logging.info(f"Model structure:\n{model.summary()}")

    return model


def train(model, network_input, network_output):
    """ 
    Train the neural network

    Args:
        model: The neural network model created with Keras
        network_input: Input sequences for the neural network
        network_output: Output sequences for the neural network

    Returns:
        None
    """
    filepath = f"data/{artist}/models/weights-improvement-{{epoch:02d}}-{{loss:.4f}}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Train the model and save the weights whenever the loss decreases
    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


def train_network():
    """ 
    Train a Neural Network to generate music 
    Args:
        None
    Returns:
        None
    """
    # Get notes from the midi files in the ./data directory
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    # prepare sequences used by the Neural Network
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # create a new model 
    model = create_network(network_input, n_vocab)

    # Check if we are starting fresh or using a previous model
    models_dir = f"data/{artist}/models/"
    weights_files = glob.glob(f"{models_dir}*.hdf5")
    if len(weights_files) > 0:
        # Load the latest weights file
        try: 
            latest_weights_file = max(weights_files, key=os.path.getctime)
            logging.info(f"Loading weights from {latest_weights_file}...")
            model.load_weights(latest_weights_file)
        except OSError: 
            logging.error(f"Unable to load weights from {weights_files[-1]}, exiting.")
            sys.exit(1)
    
    # Train the model
    train(model, network_input, network_output)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

    # Fetch the artist name from the command line arguments
    artist = sys.argv[1].lower()

    train_network()