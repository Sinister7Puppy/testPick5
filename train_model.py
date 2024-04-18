import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model
from data_preprocessing import load_and_preprocess_data
from feature_extraction import one_hot_encode_box_types, normalize_pick5_numbers
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def custom_hyperparameter_tuning(X, y, input_shape):
    """
    Perform hyperparameter tuning on the model with a custom loop.
    """
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

def main():
    try:
        csv_file_path = './pick5.csv'  # INPUT_REQUIRED {config_description: "Ensure this path points to your pick5.csv file"}
        df = load_and_preprocess_data(csv_file_path)

        if df is None:
            logging.error("Data preprocessing returned None. Check your data preprocessing steps.")
            return

        logging.info("Starting feature extraction...")
        df = one_hot_encode_box_types(df)
        df = normalize_pick5_numbers(df)
        logging.info("Feature extraction completed.")

        X = df.drop(['Date', 'Day', 'Draw'], axis=1).values
        y = df[['Pick5_0', 'Pick5_1', 'Pick5_2', 'Pick5_3', 'Pick5_4']].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

        input_shape = (X_train.shape[1], X_train.shape[2])

        logging.info("Starting hyperparameter tuning...")
        best_params = custom_hyperparameter_tuning(X_train, y_train, input_shape)
        logging.info("Hyperparameter tuning completed. Best parameters: %s", best_params)

        # Use the best parameters from hyperparameter tuning to train the model
        if best_params:
            optimizer = best_params['optimizer']
            epochs = best_params['epochs']
            batch_size = best_params['batch_size']
        else:
            # Default values if hyperparameter tuning does not return best_params
            optimizer = 'adam'
            epochs = 10
            batch_size = 64

        logging.info("Starting model training with best hyperparameters...")
        # Rebuild the model with the best optimizer from hyperparameter tuning
        model = create_model(input_shape)
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

        scores = model.evaluate(X_val, y_val, verbose=0)
        logging.info("Model evaluation completed. Mean Squared Error: %.2f, Accuracy: %.2f%%", scores[0], scores[1]*100)

        model_save_path = './models/pick5_model'
        model.save(model_save_path, save_format='tf')
        logging.info("Model saved successfully at %s in the TensorFlow SavedModel format.", model_save_path)
    except Exception as e:
        logging.error("An error occurred during model training or evaluation:", exc_info=True)

if __name__ == "__main__":
    main()