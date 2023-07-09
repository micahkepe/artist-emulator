import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import logging

# Set up logging
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Get the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the data
logging.info(f"Loading data for {artist}...")
data = np.load(f'data/{artist}/processed/processed.npz')
inputs_train = data['inputs_train']
inputs_test = data['inputs_test']
outputs_train = data['outputs_train']
outputs_test = data['outputs_test']

# Split the data into training and testing sets
logging.info("Splitting data into training and testing sets...")
inputs_train, inputs_val, outputs_train, outputs_val = train_test_split(inputs_train, outputs_train, test_size=0.2, random_state=42)

# Build the model
# Hyperparameters
logging.info("Building the model...")
number_of_labels = np.max(outputs_train) + 1
print('Number of labels:', number_of_labels)

num_epochs = 10
batch_size = 64
num_classes = number_of_labels
learning_rate = 0.001

model = Sequential()
print('Inputs Train shape:', inputs_train.shape)
model.add(LSTM(512, input_shape=(inputs_train.shape[1], inputs_train.shape[2]), return_sequences=True, recurrent_dropout=0.3))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
model.add(Dropout(0.3))
model.add(LSTM(512, recurrent_dropout=0.3))
model.add(Dense(num_classes, activation='softmax'))
logging.info(model.summary())

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
logging.info("Model compiled.")

# Define callbacks
checkpoint = ModelCheckpoint(f'data/{artist}/models/my_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.95)  # Adjust the learning rate during training
logging.info("Callbacks defined.")

# Train the model
history = model.fit(inputs_train, outputs_train, validation_data=(inputs_val, outputs_val),
                    epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping, lr_scheduler])
logging.info("Model trained.")

# Evaluate the model
model.evaluate(inputs_test, outputs_test, batch_size=batch_size)

# Save the history
np.savez(f'data/{artist}/history/history.npz', loss=history.history['loss'], val_loss=history.history['val_loss'])
logging.info("History saved.")

# Plot the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'data/{artist}/loss.png')
plt.show()
