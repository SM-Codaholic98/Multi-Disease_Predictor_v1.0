import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request
from google import genai
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import keras

# Workaround: Monkey-patch Keras layer __init__ to ignore 'quantization_config'
# This resolves errors when loading models trained in newer Keras versions.
for layer_class in [
    keras.layers.Dense, keras.layers.Dropout, keras.layers.BatchNormalization, 
    keras.layers.Activation, keras.layers.Conv1D, keras.layers.MaxPooling1D, 
    keras.layers.Flatten, keras.layers.LSTM, keras.layers.GRU, keras.layers.Bidirectional
]:
    orig_init = layer_class.__init__
    def make_safe_init(orig):
        def safe_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            orig(self, *args, **kwargs)
        return safe_init
    layer_class.__init__ = make_safe_init(orig_init)

app = Flask(__name__)

# Load Preprocessors
try:
    scaler = joblib.load('Robust-Scaler.bin')
    encoders = joblib.load('Label-Encoder.bin')
except Exception as e:
    print(f"Warning: Preprocessors not found. Ensure .bin files are present. Error: {e}")

# Target Label Mapping (Alphabetical ordering from LabelEncoder)
class_mapping = {
    0: "Diabetes",
    1: "Diabetes & Heart Disease",
    2: "Diabetes, Hypertension & Heart Disease",
    3: "Diabetes & Hypertension",
    4: "Heart Disease",
    5: "Hypertension & Heart Disease",
    6: "Hypertension",
    7: "Normal"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get Selected Model Category and Name
            model_selection = request.form['model_selection']
            model_type, model_name = model_selection.split('|')
            
            # 2. Gather form data (24 features)
            data = {
                'gender': request.form['gender'],
                'smoking': request.form['smoking'],
                'age': float(request.form['age']),
                'bmi': float(request.form['bmi']),
                'HbA1c_level': float(request.form['HbA1c_level']),
                'glucose': float(request.form['glucose']),
                'cholesterol': float(request.form['cholesterol']),
                'sleep_hours': float(request.form['sleep_hours']),
                'triglycerides': float(request.form['triglycerides']),
                'physical_activity': request.form['physical_activity'],
                'family_history': request.form['family_history'],
                'stress_level': request.form['stress_level'],
                'sugar_consumption': request.form['sugar_consumption'],
                'crp_level': float(request.form['crp_level']),
                'homocysteine_level': float(request.form['homocysteine_level']),
                'systolic_bp': float(request.form['systolic_bp']),
                'diastolic_bp': float(request.form['diastolic_bp']),
                'alcohol_intake': float(request.form['alcohol_intake']),
                'salt_intake': float(request.form['salt_intake']),
                'heart_rate': float(request.form['heart_rate']),
                'hdl': float(request.form['hdl']),
                'ldl': float(request.form['ldl']),
                'education_level': request.form['education_level'],
                'employment_status': request.form['employment_status']
            }
            
            # 3. Create DataFrame
            df = pd.DataFrame([data])
            
            # 4. Encode categorical variables
            categorical_cols = ['gender', 'smoking', 'physical_activity', 'family_history', 
                                'stress_level', 'sugar_consumption', 'education_level', 'employment_status']
            for col in categorical_cols:
                df[col] = encoders[col].transform(df[col])
                
            # 5. Scale features
            X_scaled = scaler.transform(df)
            
            # 6. Load Model and Predict
            if model_type == 'ML':
                # Machine Learning Classification Model
                model = joblib.load(f"{model_name}_Model.pkl")
                prediction_idx = int(model.predict(X_scaled)[0])
            else:
                # Deep Learning Classification Model
                model = load_model(f"{model_name}_Model.keras")
                prediction_probs = model.predict(X_scaled, verbose=0)
                prediction_idx = int(np.argmax(prediction_probs, axis=1)[0])

            # Get string label
            predicted_disease = class_mapping.get(prediction_idx, "Unknown")
            
            # 7. AI-Based Recommendation Engine using Gemini
            if predicted_disease == "Normal":
                status_class = "status-good"
                icon = "✅"
                fallback_rec = "Maintain your current healthy lifestyle and continue regular check-ups."
            elif "&" in predicted_disease or "," in predicted_disease:
                status_class = "status-danger"
                icon = "🚑"
                fallback_rec = "Immediate medical intervention and comprehensive lifestyle changes are strongly advised."
            else:
                status_class = "status-warning"
                icon = "⚠️"
                fallback_rec = "Consult a specialist for early management and targeted treatment."

            recommendation = fallback_rec
            api_key = os.environ.get("GEMINI_API_KEY", "paste yor gemini api key here")
            
            if api_key:
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
You are a professional medical AI assistant. A patient has been evaluated by a diagnostic model which predicted: {predicted_disease}.
Here is the patient's data profile:
{data}

Based on this prediction and the patient's profile, provide a highly personalized, short, professional, and actionable health recommendation (2-3 sentences max).
Do not provide any disclaimers or introductory filler, just the direct recommendation.
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    if response.text:
                        recommendation = response.text.strip()
                except Exception as e:
                    print(f"Gemini API Error: {e}")

            return render_template('predict.html', 
                                   prediction=predicted_disease, 
                                   recommendation=recommendation,
                                   status_class=status_class,
                                   icon=icon,
                                   model_name=model_name.replace('_', ' '))
                                   
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)