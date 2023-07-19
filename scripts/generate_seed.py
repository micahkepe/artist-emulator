import numpy as np
import os
import sys
import random
import logging
import datetime

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the preprocessed data
data = np.load(f'data/{artist}/preprocessed/preprocessed_data_timestamp.npz', allow_pickle=True) # change this to the latest preprocessed data file
inputs_test = data['inputs_test']
logging.info(f"Loaded preprocessed data for {artist}.")

# Randomly select a sequence from the testing data
seed_index = random.randint(0, len(inputs_test)-1)
seed = inputs_test[seed_index]
logging.info(f"Selected seed sequence from index {seed_index} of the testing data.")

# reshape the sequence to be a 3D array with 1 sequence of length 100 and 6 features
seed = np.array(seed).reshape(1, 100, 6)

# Create a directory to store the seed if it doesn't already exist
os.makedirs(f'data/{artist}/seed', exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Timestamp for versioning

# Save the seed
np.save(f'data/{artist}/seed/seed_{timestamp}.npy', seed)

logging.info(f"Seed generated and saved to data/{artist}/seed/seed_{timestamp}.npy.")
