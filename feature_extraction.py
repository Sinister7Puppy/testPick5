import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np

def one_hot_encode_box_types(df):
    """
    One-hot encodes the 'Box Type Bet' categories in the DataFrame.
    """
    if df is None:
        logging.error("DataFrame passed to one_hot_encode_box_types is None.")
        raise ValueError("DataFrame passed to one_hot_encode_box_types is None.")
    try:
        encoder = OneHotEncoder(sparse_output=False)
        box_types_encoded = encoder.fit_transform(df[['Box Type Bet']])
        # Create a DataFrame with the encoded columns
        box_types_df = pd.DataFrame(box_types_encoded, columns=encoder.get_feature_names_out(['Box Type Bet']))
        # Reset index on original DataFrame to concatenate correctly
        df.reset_index(drop=True, inplace=True)
        df_encoded = pd.concat([df, box_types_df], axis=1).drop(['Box Type Bet'], axis=1)
        logging.info("One-hot encoding of Box Type Bet categories completed successfully.")
        return df_encoded
    except Exception as e:
        logging.error("An error occurred during one-hot encoding of Box Type Bet categories:")
        logging.error(e, exc_info=True)
        raise

def normalize_pick5_numbers(df):
    """
    Normalizes the 'Pick 5' numbers into a suitable format for the neural network.
    """
    if df is None:
        logging.error("DataFrame passed to normalize_pick5_numbers is None.")
        raise ValueError("DataFrame passed to normalize_pick5_numbers is None.")
    try:
        scaler = MinMaxScaler()
        # Convert 'Pick 5' numbers to strings and then split into individual digits
        pick5_numbers = df['Pick 5'].apply(lambda x: [int(digit) for digit in str(x).zfill(5)])
        # Ensure all sequences have the same length
        if not all(len(digits) == 5 for digits in pick5_numbers):
            raise ValueError("Not all 'Pick 5' sequences have the correct length of 5 digits.")
        pick5_numbers_matrix = np.array(pick5_numbers.tolist())
        pick5_numbers_normalized = scaler.fit_transform(pick5_numbers_matrix)
        # Replace the old 'Pick 5' column with the normalized version (need to convert back to DataFrame)
        df_normalized = df.drop(['Pick 5'], axis=1)
        pick5_numbers_df = pd.DataFrame(pick5_numbers_normalized, columns=[f'Pick5_{i}' for i in range(5)])
        df_final = pd.concat([df_normalized, pick5_numbers_df], axis=1)
        logging.info("Normalization of Pick 5 numbers completed successfully.")
        return df_final
    except ValueError as ve:
        logging.error("Validation error during normalization of Pick 5 numbers:")
        logging.error(ve, exc_info=True)
        raise
    except Exception as e:
        logging.error("An error occurred during normalization of Pick 5 numbers:")
        logging.error(e, exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Load the preprocessed data
    processed_df = pd.read_csv('./processed_pick5.csv')  # INPUT_REQUIRED {config_description: "Adjust the path as necessary"}
    try:
        if processed_df.empty:
            logging.error("The processed DataFrame is empty. Please check the CSV file for correct structure and data.")
            raise ValueError("The processed DataFrame is empty.")
        df_encoded = one_hot_encode_box_types(processed_df)
        df_final = normalize_pick5_numbers(df_encoded)
        logging.info(f"Transformed DataFrame shape: {df_final.shape}")
        print(df_final.head())
    except Exception as e:
        logging.error("An error occurred in the main execution:")
        logging.error(e, exc_info=True)