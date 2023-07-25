# Artist Emulator

Artist Emulator is a project aimed at creating a deep learning model that can emulate the musical style of a given artist. The model is trained on MIDI files of the artist's compositions and learns to generate new musical sequences in a similar style.

## Project Overview

The project consists of the following components:

1. Data collection: MIDI files of the artist's compositions are collected and used as training data.

2. Data preprocessing: MIDI files are parsed and converted into sequences of musical features, such as notes, durations, intensities, offsets, and rests. These sequences are used as input for training the model.

3. Model training: A deep learning model is constructed using LSTM (Long Short-Term Memory) layers. The model is trained on the preprocessed data to learn the patterns and characteristics of the artist's music.

4. Music generation: Once the model is trained, it can be used to generate new musical sequences that emulate the style of the artist. These sequences can be converted back into MIDI files for further exploration and usage.

## Current Development Status

This project is currently under development. Here are the key points to note:

- The model training and music generation components are being actively developed and optimized.
- The current version of the model focuses on a single artist, but future updates may include support for multiple artists. 
- As of now, the dataset used as training data is not publicly available due to licensing restrictions and is only used for non-commercial purposes. If you are interested in the dataset, please contact [me](mailto:micahkepe@gmail.com).
- The project is being continuously improved to enhance the quality of the generated music and provide more customization options.

## Getting Started

To get started with the project, follow these steps:

1. Install the necessary dependencies by running `yarn`.

2. Collect MIDI files of the artist's compositions and place them in the appropriate input directory.

3. Preprocess the MIDI files by running the preprocessing script. Adjust the script parameters as needed.

4. Train the model by running the training script. Provide the artist name as a command-line argument.

5. Generate new music in the artist's style using the trained model. Customize the music generation process as desired.

## Dependencies

The project requires the following dependencies:

- `@spotify/basic-pitch`: Library for pitch estimation (`pip install @spotify/basic-pitch`)
- `music21`: Toolkit for computer-aided musicology (`pip install music21`)
- `scikit-learn`: Machine learning library for data preprocessing (`pip install scikit-learn`)
- `numpy`: Library for numerical operations (`pip install numpy`)
- `matplotlib`: Library for data visualization (`pip install matplotlib`)

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

