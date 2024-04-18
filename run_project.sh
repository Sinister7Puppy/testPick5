#!/bin/bash

# Check for Python installation and version
if ! command -v python3 &> /dev/null
then
    echo "Python could not be found. Please install Python 3."
    exit 1
fi

# Check for requirements.txt file
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt does not exist. Please ensure it is located in the project root."
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi
echo "Dependencies installed successfully."

# Train the model before running predictions
echo "Training the model..."
python3 train_model.py
if [ $? -ne 0 ]; then
    echo "Failed to train the model."
    exit 1
fi
echo "Model trained successfully."

# Run the main Python script and display output or error
echo "Running the project..."
python3 predict_next_draw.py
if [ $? -ne 0 ]; then
    echo "Failed to run the project."
    exit 1
fi
echo "Project ran successfully."