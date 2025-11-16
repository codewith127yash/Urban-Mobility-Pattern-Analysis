from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Global variables for model and label encoder
model = None
label_enc = None

# Load the model and label encoder
def load_model_and_encoder():
    global model, label_enc
    
    # Load the trained model from the specific path
    model_path = r"TrafficModel/TrafficApp/traffic_model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Load the label encoder from the specific path
    encoder_path = r"TrafficModel/TrafficApp/label_encoder.pkl"
    with open(encoder_path, 'rb') as file:
        label_enc = pickle.load(file)

# Load model and encoder when the app starts
load_model_and_encoder()

# Function to convert NumPy types to Python native types
def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        time_hour = int(request.form['hour'])
        day_of_week = int(request.form['day'])  # 0=Monday, 1=Tuesday, etc.
        car_count = int(request.form['car_count'])
        bike_count = int(request.form['bike_count'])
        bus_count = int(request.form['bus_count'])
        truck_count = int(request.form['truck_count'])
        total = car_count + bike_count + bus_count + truck_count
        
        # Prepare input for prediction
        input_data = [[time_hour, day_of_week, car_count, bike_count, bus_count, truck_count, total]]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert numerical prediction to label
        traffic_situation = label_enc.inverse_transform([prediction[0]])[0]
        
        # Fallback mapping if label encoder returns a number
        if isinstance(traffic_situation, (int, np.integer)):
            traffic_classes = ["heavy", "high", "low", "normal"]
            index = int(traffic_situation)
            if 0 <= index < len(traffic_classes):
                traffic_situation = traffic_classes[index]
        
        # Ensure it's a string for JSON serialization
        traffic_situation = str(traffic_situation)
        
        # Prepare response data
        response_data = {
            'success': True,
            'prediction': traffic_situation,
            'input_summary': {
                'hour': int(time_hour),
                'day': int(day_of_week),
                'car_count': int(car_count),
                'bike_count': int(bike_count),
                'bus_count': int(bus_count),
                'truck_count': int(truck_count),
                'total': int(total)
            }
        }
        
        # Convert any potential NumPy types to native Python types
        response_data = convert_to_native_types(response_data)
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# API route for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        
        # Get data from JSON
        time_hour = int(data['hour'])
        day_of_week = int(data['day'])
        car_count = int(data['car_count'])
        bike_count = int(data['bike_count'])
        bus_count = int(data['bus_count'])
        truck_count = int(data['truck_count'])
        total = car_count + bike_count + bus_count + truck_count
        
        # Prepare input for prediction
        input_data = [[time_hour, day_of_week, car_count, bike_count, bus_count, truck_count, total]]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert numerical prediction to label
        traffic_situation = label_enc.inverse_transform([prediction[0]])[0]
        
        # Fallback mapping if label encoder returns a number
        if isinstance(traffic_situation, (int, np.integer)):
            traffic_classes = ["heavy", "high", "low", "normal"]
            index = int(traffic_situation)
            if 0 <= index < len(traffic_classes):
                traffic_situation = traffic_classes[index]
        
        # Ensure it's a string for JSON serialization
        traffic_situation = str(traffic_situation)
        
        # Prepare response data
        response_data = {
            'success': True,
            'prediction': traffic_situation,
            'input': {
                'hour': int(time_hour),
                'day': int(day_of_week),
                'car_count': int(car_count),
                'bike_count': int(bike_count),
                'bus_count': int(bus_count),
                'truck_count': int(truck_count),
                'total': int(total)
            }
        }
        
        # Convert any potential NumPy types to native Python types
        response_data = convert_to_native_types(response_data)
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Data visualization route
@app.route('/data')
def data_visualization():
    return render_template('data_visualization.html')

if __name__ == '__main__':
    app.run(debug=False)
