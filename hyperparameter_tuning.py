import logging
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from model import create_model
from data_preprocessing import load_and_preprocess_data
from feature_extraction import one_hot_encode_box_types, normalize_pick5_numbers

logging.basicConfig(level=logging.INFO)

def custom_hyperparameter_tuning(X, y, input_shape):
    """
    Perform hyperparameter tuning on the model with a custom loop.
    """
    try:
        best_score = float('-inf')
        best_params = {}
        for batch_size in [16, 32, 64]:
            for epochs in [50, 100]:
                for optimizer in ['adam', 'rmsprop']:
                    logging.info(f"Training with batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}")
                    model = create_model(input_shape=input_shape)
                    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
                    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
                    score = np.mean(history.history['val_accuracy'][-5:])  # Average of last 5 epochs
                    logging.info(f"Validation accuracy: {score}")
                    if score > best_score:
                        best_score = score
                        best_params = {'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer}
                        logging.info(f"New best score: {best_score} with params {best_params}")
        logging.info(f"Best score: {best_score} using {best_params}")
        return best_params
    except Exception as e:
        logging.error("An error occurred during hyperparameter tuning:", exc_info=True)
        raise

def main():
    try:
        csv_file_path = './pick5.csv'  # INPUT_REQUIRED {config_description: "Ensure this path points to your pick5.csv file"}
        df = load_and_preprocess_data(csv_file_path)

        if df is None:
            logging.error("Data preprocessing returned None. Check your data preprocessing steps.")
            return

        df = one_hot_encode_box_types(df)
        df = normalize_pick5_numbers(df)

        X = df.drop(['Date', 'Day', 'Draw', 'Box Type Bet'], axis=1).values
        y = df[['Pick5_0', 'Pick5_1', 'Pick5_2', 'Pick5_3', 'Pick5_4']].values

        # Split the data into training and validation sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        input_shape = (1, X_train.shape[2])

        best_params = custom_hyperparameter_tuning(X_train, y_train, input_shape)
        logging.info("Best hyperparameters found: %s", best_params)
    except Exception as e:
        logging.error("An error occurred in the main execution:", exc_info=True)

if __name__ == "__main__":
    main()