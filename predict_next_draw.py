import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from data_preprocessing import load_and_preprocess_data
from feature_extraction import one_hot_encode_box_types, normalize_pick5_numbers

logging.basicConfig(level=logging.INFO)

def load_model(model_path='./models/pick5_model'):
    """Load the trained model from the specified path."""
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully from %s.", model_path)
        return model
    except Exception as e:
        logging.error("Failed to load model from %s.", model_path)
        logging.error(e, exc_info=True)
        return None

def predict_next_draw(model, latest_draw_data):
    """Predict the next draw numbers using the loaded model."""
    try:
        # Preprocess the input data to match the training data format
        preprocessed_data = load_and_preprocess_data(latest_draw_data)
        if preprocessed_data is None or preprocessed_data.empty:
            logging.error("Preprocessed data is None or empty. Cannot proceed with prediction.")
            return None
        df_encoded = one_hot_encode_box_types(preprocessed_data)
        if df_encoded is None:
            logging.error("Failed to encode box types. Cannot proceed with prediction.")
            return None
        df_final = normalize_pick5_numbers(df_encoded)
        if df_final is None:
            logging.error("Failed to normalize Pick 5 numbers. Cannot proceed with prediction.")
            return None
        
        # Assuming the latest draw data is a single row DataFrame
        X_pred = df_final.drop(['Date', 'Day', 'Draw'], axis=1).values
        X_pred = np.reshape(X_pred, (1, 1, X_pred.shape[1]))  # Reshape for the model
        
        # Predict the next draw
        prediction = model.predict(X_pred)
        return prediction
    except Exception as e:
        logging.error("Failed to predict the next draw.")
        logging.error(e, exc_info=True)
        return None

if __name__ == "__main__":
    # Path to the latest draw data, this needs to be updated with the actual latest draw data file
    latest_draw_data = './processed_pick5.csv'  # INPUT_REQUIRED {config_description: "This should be the path to the latest processed draw data"}
    model = load_model()
    if model:
        prediction = predict_next_draw(model, latest_draw_data)
        if prediction is not None:
            # Convert prediction to actual numbers and calculate accuracy (placeholder for demonstration)
            predicted_numbers = np.round(prediction).astype(int).flatten()
            print(f"Predicted Winning Number - {' - '.join(map(str, predicted_numbers))}")
            # Note: Accuracy percentages are placeholders. Implement actual accuracy calculation based on model output.
            print("Digit n accuracy - Placeholder%")
        else:
            logging.error("Prediction failed.")
    else:
        logging.error("Model could not be loaded. Prediction aborted.")