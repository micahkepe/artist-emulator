import numpy as np
import os
import sys
import random

# Fetch the artist name from the command line arguments
artist = sys.argv[1].lower()

# Load the preprocessed data
data = np.load(f'data/{artist}/processed/processed.npz', allow_pickle=True)
inputs_test = data['inputs_test']

# Randomly select a sequence from the testing data
seed_index = random.randint(0, len(inputs_test)-1)
seed = inputs_test[seed_index]

# Create a directory to store the seed if it doesn't already exist
os.makedirs(f'data/{artist}/seed', exist_ok=True)

# Save the seed
np.save(f'data/{artist}/seed/seed.npy', seed)

print(f"Seed generated and saved to data/{artist}/seed/seed.npy.")
