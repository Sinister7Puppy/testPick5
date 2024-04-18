import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

# Suppress TensorFlow warnings and set to use CPU if CUDA not available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if not tf.config.list_physical_devices('GPU'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logging.basicConfig(level=logging.INFO)

def create_model(input_shape, optimizer='adam', loss='mse', metrics=['accuracy']):
    logging.info("Creating model with adjusted input shape for Box Type Bet categories: %s", input_shape)
    try:
        # Define a Sequential model
        model = keras.Sequential()

        # Adjusting the input shape to accommodate one-hot encoded Box Type Bet categories
        # The new input_shape must account for the additional dimensions added by one-hot encoding
        # For example, if there are 6 Box Type Bet categories, and each category is one-hot encoded,
        # the input_shape's second dimension (feature dimension) will increase by 6.

        # Add a Dense layer to process the one-hot encoded vectors before the LSTM layers.
        # This is necessary because the LSTM layers are better at handling sequential data,
        # and we need to initially process the one-hot encoded categorical data which is not sequential.
        model.add(keras.Input(shape=input_shape))
        logging.info("Added Input layer with shape adjusted for one-hot encoded Box Type Bet categories.")

        # Adding a Dense layer to handle the one-hot encoded vectors
        model.add(layers.Dense(64, activation='relu'))
        logging.info("Added Dense layer to process one-hot encoded vectors.")

        # Add an LSTM layer with 128 units. Adjust the units based on dataset size and complexity
        model.add(layers.LSTM(128, return_sequences=True))
        logging.info("Added LSTM layer with 128 units.")
        # Adding dropout to prevent overfitting
        model.add(layers.Dropout(0.2))
        logging.info("Added Dropout layer.")

        # Adding a second LSTM layer, making sure to return sequences if you plan to add more LSTM layers after this
        model.add(layers.LSTM(64, return_sequences=False))
        logging.info("Added second LSTM layer with 64 units.")
        # Another dropout layer
        model.add(layers.Dropout(0.2))
        logging.info("Added second Dropout layer.")

        # Dense layer for output prediction
        model.add(layers.Dense(5, activation='linear'))  # 5 units for the 5 digits in the Pick5Lotto
        logging.info("Added Dense output layer with 5 units.")

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logging.info(f"Compiled the model with {optimizer} optimizer and {loss} loss.")

        # Summary of the model
        model.summary()

        return model
    except Exception as e:
        logging.error("Error creating the model: %s", e, exc_info=True)
        raise e

if __name__ == "__main__":
    # Adjust the input_shape based on your dataset's sequence length, features, and the dimensionality of one-hot encoded vectors
    input_shape = (None, 5 + 6)  # +6 for the one-hot encoded Box Type Bet categories, adjust as necessary
    # INPUT_REQUIRED {config_description} - Adjust the input_shape based on your dataset's sequence length and features
    try:
        model = create_model(input_shape)
    except Exception as e:
        logging.error("Error in model initialization: %s", e, exc_info=True)