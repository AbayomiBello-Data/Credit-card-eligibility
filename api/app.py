import streamlit as st
import pandas as pd
import openai
import os
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'model/saved_models/model.pkl')  # Path to the saved model (default is 'model.pkl')
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        logging.info(f"Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Set up session state for the chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant about credit scores and card usage."}]

# Helper function to encode categorical variables
def encode_input(gender, income_type, education, family_status, housing_type):
    return {
        "Applicant_Gender_Encoded": {"Male": 1, "Female": 0}[gender],
        "Income_Type_Encoded": {"Working": 0, "Commercial associate": 1, "Pensioner": 2, "State servant": 3, "Student": 4}[income_type],
        "Education_Type_Encoded": {"Secondary / secondary special": 0, "Higher education": 1, "Incomplete higher": 2, "Lower secondary": 3, "Academic degree": 4}[education],
        "Family_Status_Encoded": {"Married": 0, "Single / not married": 1, "Civil marriage": 2, "Separated": 3, "Widow": 4}[family_status],
        "Housing_Type_Encoded": {"House / apartment": 0, "With parents": 1, "Municipal apartment": 2, "Rented apartment": 3, "Office apartment": 4, "Co-op apartment": 5}[housing_type],
    }

# --- Credit Card Eligibility Form ---
col_main, col_chat = st.columns([2, 1])
with col_main:
    st.title("💳 Credit Card Eligibility Checker")
    with st.form("eligibility_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
            education = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
            family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
            housing_type = st.selectbox("Housing Type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
            car = st.radio("Owns Car?", ["Yes", "No"])
            realty = st.radio("Owns Realty?", ["Yes", "No"])
        with col2:
            income = st.number_input("Total Income", min_value=0.0)
            age = st.number_input("Age", min_value=18)
            work_years = st.number_input("Years of Working", min_value=0.0)
            children = st.number_input("Total Children", min_value=0)
            fam_members = st.number_input("Total Family Members", min_value=1)
            good_debt = st.number_input("Good Debt", min_value=0.0)
            bad_debt = st.number_input("Bad Debt", min_value=0.0)

        submitted = st.form_submit_button("Check Eligibility")
        if submitted and model:
            try:
                # Prepare input data
                encoded_data = encode_input(gender, income_type, education, family_status, housing_type)
                data = {
                    **encoded_data,
                    "Total_Income": income,
                    "Applicant_Age": age,
                    "Years_of_Working": work_years,
                    "Total_Children": children,
                    "Total_Family_Members": fam_members,
                    "Total_Bad_Debt": bad_debt,
                    "Total_Good_Debt": good_debt,
                    "Owned_Car": 1 if car == "Yes" else 0,
                    "Owned_Realty": 1 if realty == "Yes" else 0
                }
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                st.success(f"Prediction Score: {prediction:.2f}")
                if prediction > 0.5:
                    st.success("✅ Eligible for Credit Card")
                else:
                    st.warning("❌ Not Eligible for Credit Card")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- Chat Assistant ---
with col_chat:
    st.title("💬 Chat Assistant")
    st.markdown("Ask me anything about credit cards, scores, and tips.")
    user_input = st.chat_input("Type your question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.container():
            full_response = ""
            message_placeholder = st.empty()

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages,
                    temperature=0.7,
                    stream=True
                )

                for chunk in response:
                    if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                        full_response += chunk["choices"][0]["delta"]["content"]
                        message_placeholder.markdown(
                            f"<div class='bot-bubble'><strong>Assistant:</strong><br>{full_response}</div>", unsafe_allow_html=True
                        )
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"OpenAI API Error: {e}")

    # Display chat history
    for msg in st.session_state.messages[1:]:
        role = msg["role"]
        content = msg["content"]
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"

        st.markdown(
            f"<div class='{bubble_class}'><strong>{'You' if role == 'user' else 'Assistant'}:</strong><br>{content}</div>",
            unsafe_allow_html=True
        )

# 💬 Bubble styling
st.markdown(""" 
<style>
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 15px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 15px;
    margin: 8px 0;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)
