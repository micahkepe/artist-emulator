# Artist Emulator

Artist Emulator is a project aimed at creating a deep learning model that can emulate the musical style of a given artist. The model is trained on MIDI files of the artist's compositions and learns to generate new musical sequences in a similar style.

## Project Overview

The project consists of the following components:

1. Data collection: MIDI files of the artist's compositions are collected and used as training data.

2. Data preprocessing: MIDI files are parsed and converted into sequences of musical features, such as notes, durations, intensities, offsets, and rests. These sequences are used as input for training the model.

3. Model training: A deep learning model is constructed using LSTM (Long Short-Term Memory) layers. The model is trained on the preprocessed data to learn the patterns and characteristics of the artist's music.

4. Saving model weights: After training, the latest model weights are saved, enabling users to resume training or use the model for generation without retraining.

5. Music generation: Once the model is trained, it can be used to generate new musical sequences that emulate the style of the artist. These sequences can be converted back into MIDI files for further exploration and usage.

## Current Development Status

This project is currently under development. Here are the key points to note:

- The model training and music generation components are being actively developed and optimized.
- The current version of the model focuses on a single artist, Bach, but future updates may include support for multiple artists. 
- The project is being continuously improved to enhance the quality of the generated music and provide more customization options.

## Getting Started

To get started with the project, follow these steps:

1. Install the necessary dependencies by running `pip install -r requirements.txt`.

2. Collect MIDI and/or MP3 files of the artist's compositions and place them in an appropriate input directory.

3. Process any MP3 files of the artist's compositions into MIDI files using the `convert_to_midi.py` script. Provide the input and output directories as command-line arguments.

4. Train the model by running the `model_creation.py` script. Change the specified paths in the script to match your local directories. Adjust the hyperparameters and architecture as desired. After training, the latest model weights will be saved in the specified directory.

5. Generate new music in the artist's style using the trained model with the `predict.py` script. Customize the music generation process as desired. Again, change the specified paths in the script to match your local directories.

## Dependencies

The project requires the following dependencies:

- `@spotify/basic-pitch`: Library for pitch estimation (`pip install @spotify/basic-pitch`)
- `music21`: Toolkit for computer-aided musicology (`pip install music21`)
- `scikit-learn`: Machine learning library for data preprocessing (`pip install scikit-learn`)
- `numpy`: Library for numerical operations (`pip install numpy`)
- `matplotlib`: Library for data visualization (`pip install matplotlib`)
- `h5py`: Library to save and load model weights in the HDF5 format (`pip install h5py`)

## Contributions

Contributions to this project are welcome! If you find any issues or have ideas for improvement, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code as per the terms of the license.


## References

Some websites that I found helpful in understanding the LSTM model are listed below:
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
- [RNN Wikipedia Page](https://en.wikipedia.org/wiki/Recurrent_neural_network)
- [LTSM Wikipedia Page](https://en.wikipedia.org/wiki/Long_short-term_memory)

Additionally, this project uses insights from the article ["How to generate music using a LSTM neural network in Keras"](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5) by Sigurður Skúli. 

## Contact

For any questions or inquiries, please contact [me](mailto:micahkepe@.com).

