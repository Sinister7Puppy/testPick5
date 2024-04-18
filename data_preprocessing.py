import pandas as pd
from collections import Counter
import logging

def categorize_box_type_bet(draw):
    """
    Categorizes the draw into one of the six Box Type Bets based on the uniqueness and repetition of numbers.
    :param draw: A string representing the drawn numbers, e.g., "12345"
    :return: A string representing the category of Box Type Bet.
    """
    counts = Counter(draw)
    unique_counts = list(counts.values())

    if len(counts) == 5:
        return '120-WAY'
    elif len(counts) == 4 and unique_counts.count(2) == 1:
        return '60-WAY'
    elif len(counts) == 3:
        if unique_counts.count(3) == 1:
            return '20-WAY'
        elif unique_counts.count(2) == 2:
            return '30-WAY'
    elif len(counts) == 2:
        if unique_counts.count(4) == 1:
            return '5-WAY'
        elif unique_counts.count(3) == 1 and unique_counts.count(2) == 1:
            return '10-WAY'
    else:
        return 'Unknown'

def load_and_preprocess_data(csv_file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Check if 'Day - Draw' column exists
        if 'Day - Draw' not in df.columns:
            logging.error("'Day - Draw' column is missing from the CSV file. Please ensure the CSV file has the correct structure.")
            raise ValueError("'Day - Draw' column is missing from the CSV file.")

        # Parse the 'Date' column to datetime format, handling inconsistent formatting
        df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce', format='%m/%d/%Y')
        
        # Drop rows where 'Date' is NaT due to parsing errors
        df = df.dropna(subset=['Date'])

        # Split the 'Day - Draw' column into two separate columns ('Day' and 'Draw')
        df[['Day', 'Draw']] = df['Day - Draw'].str.split(' - ', expand=True)

        # Validate and format the 'Pick 5' numbers to ensure consistent formatting
        # Assuming 'Pick 5' numbers are integers in the CSV, convert them to a string with leading zeros
        df['Pick 5'] = df['Pick 5'].apply(lambda x: f"{x:05d}")

        # Categorize each draw into one of the six Box Type Bets
        df['Box Type Bet'] = df['Pick 5'].apply(lambda x: categorize_box_type_bet(x))

        # Drop the original 'Day - Draw' column as it's no longer needed
        df.drop('Day - Draw', axis=1, inplace=True)

        logging.info("Data preprocessing completed successfully.")
        # Save the preprocessed DataFrame to a CSV file
        df.to_csv('./processed_pick5.csv', index=False)
        logging.info("Saving preprocessed data to './processed_pick5.csv'")
        return df
    except Exception as e:
        logging.error("An error occurred during data preprocessing:")
        logging.error(e, exc_info=True)
        return pd.DataFrame()  # Return an empty DataFrame to avoid 'NoneType' errors in subsequent processing

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage, adjust the CSV file path as needed
    csv_path = './pick5.csv'  # INPUT_REQUIRED {config_description: "Adjust the CSV file path as needed"}
    try:
        processed_df = load_and_preprocess_data(csv_path)
        if not processed_df.empty:
            print(processed_df.head())
        else:
            logging.error("Processed DataFrame is empty. Please check the CSV file for correct structure and data.")
    except Exception as e:
        logging.error("An error occurred in the main execution:")
        logging.error(e, exc_info=True)