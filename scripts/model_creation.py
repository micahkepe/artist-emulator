""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import to_categorical as np_utils
from keras.callbacks import ModelCheckpoint
import logging
import os

def train_network():
    """
    Train a Neural Network to generate music

    Args:
        None

    Returns:
        None 
    """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ 
    Get all the notes and chords from the midi files in the right directory

    Args:
        None

    Returns:
        notes (list): All the notes and chords from the midi files
    """
    notes = []

    # check if notes have already been saved to file
    if os.path.exists('data/bach/data/notes'):
        if os.path.getsize('data/bach/data/notes') > 0:
            logging.info("Loading notes from data/notes")
            with open('data/bach/data/notes', 'rb') as filepath:
                notes = pickle.load(filepath)
            logging.info("Notes loaded from data/notes")
            return notes
    
    for file in glob.glob("data/bach/midi/*.mid"):
        midi = converter.parse(file)

        logging.info("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/bach/data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
        logging.info(f"Notes saved to data/notes")

    return notes

def prepare_sequences(notes, n_vocab):
    """
    Prepare the sequences used by the Neural Network
    
    Args:
        notes (list): The notes from the midi files
        n_vocab (int): The number of unique notes
        
    Returns:
        network_input (list): The input sequences for the Neural Network
        network_output (list): The output sequences for the Neural Network
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

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """
    Create the structure of the neural network

    Args:
        network_input (list): The input sequences for the Neural Network
        n_vocab (int): The number of unique notes

    Returns:
        model (keras.model): The Keras model of the neural network
    """
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

    return model

def train(model, network_input, network_output):
    """ 
    Train the neural network

    Args:
        model (keras.model): The Keras model of the neural network
        network_input (list): The input sequences for the Neural Network
        network_output (list): The output sequences for the Neural Network

    Returns:
        None
    """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)
    train_network()