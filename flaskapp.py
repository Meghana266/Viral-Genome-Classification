from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
import os
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import csv
import io

app = Flask(__name__)
output_folder = "CNN_Model_Output"
model_path = os.path.join(output_folder, "cnn_model_all.h5")
model = load_model(model_path)

# Define a dictionary to map folder names to disease names and labels
disease_mapping = {
    0: "HBV",
    1: "INFLUENZA",
    2: "HCV",
    3: "DENGUE"
}

# Define reference sequences for diseases
reference_sequences = {
    "HBV": "HBV_REFERENCE_SEQUENCE",
    "INFLUENZA": "INFLUENZA_REFERENCE_SEQUENCE",
    "HCV": "HCV_REFERENCE_SEQUENCE",
    "DENGUE": "DENGUE_REFERENCE_SEQUENCE"
}


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded_sequence = sequence + 'N' * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        padded_sequences.append(padded_sequence)
    return padded_sequences

def one_hot_encoding(seq):
    base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    return np.array([base_dict.get(base, [0, 0, 0, 0]) for base in seq])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']
    file = request.files.get('file')  # Use .get() to avoid KeyError
    predictions = []

    if file and file.filename:
        # Process CSV file
        data = file.read().decode('utf-8').splitlines()
        for row in csv.reader(data):
            seq = row[0]
            max_length = 11195
            padded_sequence = pad_sequences([seq], max_length)
            encoded_sequence = one_hot_encoding(padded_sequence[0])
            input_data = np.array(encoded_sequence).reshape(1, max_length, 4)

            # Make prediction
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_disease = disease_mapping.get(predicted_class)
            predictions.append([seq, predicted_disease])

    else:
        # Process single sequence
        if sequence:
            max_length = 11195
            padded_sequence = pad_sequences([sequence], max_length)
            encoded_sequence = one_hot_encoding(padded_sequence[0])
            input_data = np.array(encoded_sequence).reshape(1, max_length, 4)

            # Make prediction
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_disease = disease_mapping.get(predicted_class)
            predictions.append([sequence, predicted_disease])
        else:
            return jsonify({"error": "No sequence or file provided."})

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
