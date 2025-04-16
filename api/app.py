from flask import Flask, request, jsonify, render_template
import openai
import mlflow
import pandas as pd
import os
import logging
from mlflow.pyfunc import load_model
from flask_basicauth import BasicAuth
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('AUTH_USERNAME', 'admin')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('AUTH_PASSWORD', 'password')

basic_auth = BasicAuth(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
MODEL_URI = os.getenv('MODEL_URI', 'models:/fraud_detection/Production')
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'


# Set your OpenAI key in environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/chat', methods=['POST'])
@basic_auth.required
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who specializes in credit card approval, security tips,"
                "fraud prevention, and improving eligibility for financial products. "
                "Provide clear and friendly answers to help users better understand how to get approved "
                "and use credit cards responsibly."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
          



# Load the ML model
try:
    model = load_model(MODEL_URI)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.form.to_dict()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Binary Yes/No fields
        yes_no_fields = [
            'Owned_Car', 'Owned_Realty', 'Owned_Mobile_Phone',
            'Owned_Work_Phone', 'Owned_Phone', 'Owned_Email'
        ]

        for field in yes_no_fields:
            data[field] = 1 if data[field].strip().lower() == 'yes' else 0

        # Convert numeric fields to float
        numeric_fields = [
            'Total_Children', 'Total_Income', 'Total_Family_Members',
            'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt', 'Total_Good_Debt'
        ]
        for field in numeric_fields:
            data[field] = float(data[field])

        # Label encoding of categorical values (manually mapped)
        # These mappings MUST match what the LabelEncoder saw during training

        gender_map = {'Male': 1, 'Female': 0}
        income_map = {
            'Working': 0, 'Commercial associate': 1, 'Pensioner': 2,
            'State servant': 3, 'Student': 4
        }
        education_map = {
            'Secondary / secondary special': 0,
            'Higher education': 1,
            'Incomplete higher': 2,
            'Lower secondary': 3,
            'Academic degree': 4
        }
        family_status_map = {
            'Married': 0,
            'Single / not married': 1,
            'Civil marriage': 2,
            'Separated': 3,
            'Widow': 4
        }
        housing_map = {
            'House / apartment': 0,
            'With parents': 1,
            'Municipal apartment': 2,
            'Rented apartment': 3,
            'Office apartment': 4,
            'Co-op apartment': 5
        }

        data['Applicant_Gender_Encoded'] = gender_map.get(data['Applicant_Gender_Encoded'], -1)
        data['Income_Type_Encoded'] = income_map.get(data['Income_Type_Encoded'], -1)
        data['Education_Type_Encoded'] = education_map.get(data['Education_Type_Encoded'], -1)
        data['Family_Status_Encoded'] = family_status_map.get(data['Family_Status_Encoded'], -1)
        data['Housing_Type_Encoded'] = housing_map.get(data['Housing_Type_Encoded'], -1)

        # Check for any failed mapping
        if -1 in (
            data['Applicant_Gender_Encoded'],
            data['Income_Type_Encoded'],
            data['Education_Type_Encoded'],
            data['Family_Status_Encoded'],
            data['Housing_Type_Encoded']
        ):
            return jsonify({'error': 'Invalid categorical input'}), 400

        # Prepare DataFrame
        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]
        is_eligible = prediction > 0.5

        logging.info(f"User input: {data}")
        logging.info(f"Prediction: {prediction} | Eligible: {is_eligible}")

        return jsonify({
            'prediction': float(prediction),
            'is_eligible': bool(is_eligible)
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(SERVER_PORT), debug=DEBUG_MODE)
