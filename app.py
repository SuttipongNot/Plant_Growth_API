from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("plant_growth.pkl")

@app.route('/api/plant_growth', methods=['POST'])
def plant_growth():
    Soil_Type = int(request.form.get('Soil_Type')) 
    Sunlight_Hours = float(request.form.get('Sunlight_Hours')) 
    Water_Frequency = int(request.form.get('Water_Frequency')) 
    Fertilizer_Type = int(request.form.get('Fertilizer_Type')) 
    Temperature = float(request.form.get('Temperature')) 
    Humidity = float(request.form.get('Humidity'))

    
    # Prepare the input for the model
    x = np.array([[Soil_Type, Sunlight_Hours, Water_Frequency,Fertilizer_Type,Temperature,Humidity]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    return {'Growth_Milestone': int(prediction[0])}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)