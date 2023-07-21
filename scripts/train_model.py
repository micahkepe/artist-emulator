import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import logging
import datetime

# Set up logging
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the preprocessed data
logging.info(f"Loading data for {artist}...")
data = np.load(f'data/{artist}/preprocessed/preprocessed_data_1689734573.npz', allow_pickle=True) # change this to the latest preprocessed data file
inputs_train = data['inputs_train']
inputs_test = data['inputs_test']
outputs_train = data['outputs_train']
outputs_test = data['outputs_test']

# Load the note encoder
note_encoder = LabelEncoder()
note_encoder.classes_ = np.load(f'data/{artist}/preprocessed/note_encoder_1689734573.npy') # change this to the latest processed data file

# Start building the model
# Hyperparameters
logging.info("Building the model...")
number_of_labels = len(note_encoder.classes_)  # Number of labels is equal to the number of unique notes, chords, and rests
num_epochs = 100
batch_size = 64
initial_learning_rate = 0.001

model = Sequential()
model.add(LSTM(256, input_shape=(inputs_train.shape[1], inputs_train.shape[2]), return_sequences=True, dropout=0.3))  # Decreased number of neurons
model.add(LSTM(256, return_sequences=True, dropout=0.3))  # Decreased number of neurons
model.add(LSTM(256, dropout=0.3))  # Decreased number of neurons
model.add(Dense(number_of_labels, activation='softmax'))  # Softmax activation for multi-class classification
logging.info(f"Model structure:\n{model.summary()}")

# Learning rate decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])  # Use categorical_crossentropy for multi-class classification
logging.info("Model compilation completed.")

# Define callbacks
checkpoint = ModelCheckpoint(f'data/{artist}/models/model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
callbacks_list = [checkpoint, early_stopping]
logging.info("Callbacks defined.")

# Train the model
history = model.fit(inputs_train, outputs_train, validation_split=0.2,  # Use validation_split instead of manually splitting data
                    epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1)
logging.info("Model training completed.")

# Evaluate the model
model.evaluate(inputs_test, outputs_test, batch_size=batch_size)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Timestamp for versioning

# Save the model
os.makedirs(f'data/{artist}/models', exist_ok=True)
model.save(f'data/{artist}/models/model_{timestamp}.h5')
logging.info(f"Model saved to data/{artist}/models/model_{timestamp}.h5")

# Save the history
os.makedirs(f'data/{artist}/history', exist_ok=True)
np.savez(f'data/{artist}/history/history_{timestamp}.npz', loss=history.history['loss'], val_loss=history.history['val_loss'])
logging.info("History saved.")

# Plot the loss
plt.figure()
plt.plot(history.history['loss'], 'b', label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.plot(history.history['accuracy'], 'g', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'y', label='Validation accuracy')
plt.title('Training and validation loss and accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss and Accuracy')
plt.legend()
plt.savefig(f'data/{artist}/loss.png')
plt.show()