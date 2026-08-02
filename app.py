import streamlit as st
import joblib
import pandas as pd
import numpy as np 
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import os
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------------
# Load ML Files
# -------------------------------
premium_model = joblib.load("models/gradient_boosting_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# Fraud Detection Model
# -------------------------------
fraud_model = joblib.load("models/xgboost_fraud_model.pkl")
fraud_preprocessor = joblib.load("models/fraud_preprocessor.pkl")
#--------------------------------
# Car Damage Model
#--------------------------------
damage_model = load_model("models/mobilenetv2_car_damage_detection.keras")
#----------------------------------
# Review Sentiment Model 
#---------------------------------

sentiment_model = load_model("models/lstm_sentiment_model.keras")

tokenizer = joblib.load("models/tokenizer.pkl")
#------------------------------------------------
# ----------------------------------------------

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

chat_model = genai.GenerativeModel("gemini-flash-latest")

#------------------------------------------------------------


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="InsureAI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# Session State
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# LOGIN PAGE
# -------------------------------
if not st.session_state.logged_in:

    st.markdown(
        "<h1 style='text-align:center;'>🛡️ InsureAI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;color:gray;'>CAPSTONE DEMO CONSOLE</h3>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("🔐 Sign in to Console", use_container_width=True):

        if username and password:
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.warning("Please enter Username and Password.")

# -------------------------------
# HOME PAGE
# -------------------------------
else:

    # Sidebar
    st.sidebar.title("🛡️ InsureAI")
    st.sidebar.success("Insurance Claim Intelligence Platform")

    module = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Home",
            "💰 Premium Prediction",
            "🚨 Fraud Detection",
            "🚗 Damage Detection",
            "😊 Review Sentiment",
            "🤖 InsureAI Chatbot"
        ]
    )

    st.sidebar.write("---")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- HOME ----------------
    if module == "🏠 Home":

        st.title("🏠 Welcome to InsureAI")

        st.success("Insurance Claim Intelligence Platform")

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.info("💰 Premium Prediction")
            st.write("Predict insurance premium using Machine Learning.")

            st.info("🚨 Fraud Detection")
            st.write("Detect fraudulent insurance claims.")

            st.info("🚗 Damage Detection")
            st.write("Estimate vehicle damage using AI.")

        with col2:
            st.info("😊 Review Sentiment")
            st.write("Analyze customer review sentiment.")

            st.info("🤖 InsureAI Chatbot")
            st.write("Ask insurance-related questions using Gemini AI.")

    # ---------------- PREMIUM ----------------
    
    
    elif module == "💰 Premium Prediction":

        st.title("💰 Insurance Premium Prediction")

        st.write("Enter customer information below.")

        col1, col2 = st.columns(2)

        with col1:

         age = st.number_input("Age", 18, 100, 30)

         gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            50.0,
            25.0
        )

        children = st.slider(
            "Number of Children",
            0,
            5,
            0
        )

        with col2:

          smoker = st.selectbox(
            "Smoker",
            ["no", "yes"]
        )

          region = st.selectbox(
            "Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ])
        occupation = st.selectbox(
            "Occupation",
            [
               "business_owner",
                "retired",
                "salaried",
                "self_employed",
                "student"
            ])
        annual_income = st.number_input(
          "Annual Income (INR)",
           min_value=0,
           value=500000,
           step=10000
        )
        exercise = st.selectbox(
         "Exercise Frequency",
        [
           "none",
            "1-2_per_week",
            "3-4_per_week",
             "daily"
        ]
        )
        alcohol = st.selectbox(
          "Alcohol Consumption",
        [
           "none",
            "occasional",
            "regular"
        ]

        )
        medical_history = st.selectbox(
          "Medical History",
         [
           "none",
            "diabetes",
             "hypertension",
            "diabetes_hypertension",
             "heart_disease"
         ]
         )
        family_history = st.selectbox(
        "Family Medical History",
        [
          "no",
          "yes"
        ]
        )
      

        st.write("")
        st.write("")

        predict = st.button(
            "💰 Predict Premium",
            use_container_width=True
        )

        st.write("---")

        if predict:

        # Create input DataFrame
         input_df = pd.DataFrame({
          "age": [age],
           "gender": [gender],
           "bmi": [bmi],
           "children": [children],
            "smoker": [smoker],
            "region": [region],
            "occupation": [occupation],
            "annual_income_inr": [annual_income],
            "exercise_frequency": [exercise],
            "alcohol_consumption": [alcohol],
             "medical_history": [medical_history],
             "family_medical_history": [family_history]
    })

        # Apply preprocessing
         encoded = preprocessor.transform(input_df)

        # Apply scaling
         scaled = scaler.transform(encoded)

        # Predict
         prediction = premium_model.predict(scaled)

         st.success("Prediction Completed Successfully!")

         st.metric(
         "Predicted Insurance Premium",
        f"₹ {prediction[0]:,.2f}"
    )
        

   # ---------------- FRAUD ----------------
    elif module == "🚨 Fraud Detection":

      st.title("🚨 Insurence Fraud Detection")

      st.write("Enter claims detai below.")
      col1, col2 = st.columns(2)

      with col1:

         customer_age= st.number_input(
            "Age",
            18,
            100,
            35,
            key="fraud_age"
        )
         gender = st.selectbox(
            "Gender",
            ["male", "female"],
            key="fraud_gender"
        )
         annual_premium = st.number_input(
            "Annual Premium (INR)",
            min_value=0,
            value=500000,
            step=10000,
            key="fraud_income"
        )
         policy_tenure = st.number_input(
            "Policy Tenure (Years)",
            min_value=0,
            max_value=50,
            value=5
        )
         days_to_report = st.number_input(
           "Days to Report",
            min_value=0,
            max_value=365,
            value=5
        )
         policy_type = st.selectbox(
         "Policy Type",
        [
           "health",
          " motor",
           "property"
        ]
        )

         incident_type = st.selectbox(
          "Incident Type",
        [
            "critical_illness",
            "hospitalization",
            "collision",
            "natural_disaster",
            "third_party_damage",
            "surgery",
            "theft",
            "fire",
            "burglary"
        ]
        )

         incident_severity = st.selectbox(
          "Incident Severity",
           [
            "minor",
            "moderate",
             "major",
             "total_loss"
            ]
            )
      with col2:

             claim_amount = st.number_input(
             "Claim Amount",
               min_value=0,
               value=50000,
               step=1000,
               key="fraud_claim"
            )
             witnesses = st.number_input(
             "Number of Witnesses",
             min_value=0,
             max_value=10,
             value=1
           )
             past_claims = st.number_input(
              "Past Claims (Last 3 Years)",
               min_value=0,
                max_value=20,
                value=0
            )
             police_report = st.selectbox(
                "Police Report Filed",
             [
                "yes",
                  "no",
                  "not_applicable"
             ]
            )
             documents_complete = st.selectbox(

              
               "Documents Complete",
            [
                    "yes",
                    "no"
             ]
              )
             

             claim_channel = st.selectbox(
              "Claim Channel",
               [
                    "online",
                      "agent",
                     "branch"
                ]
                ) 

             income_bracket = st.selectbox(
                "Income Bracket",
                [
                   "<3L",
                   "3L-6L",
                    "6L-12L",
                     ">12L"
                ]
              )

             claim_location = st.selectbox(
                  "Claim Location",
                      [
                      "urban",
                      "semi_urban",
                       "rural"
                       ]
)
             

             st.write("")
             st.write("")

             fraud_predict = st.button(
            "🚨 Detect Fraud",
            use_container_width=True
             )
             if fraud_predict:

                input_df = pd.DataFrame({
                "customer_age":[customer_age],
                 "gender":[gender],
                 "policy_type":[policy_type],
                "policy_tenure_years":[policy_tenure],
                 "annual_premium_inr":[annual_premium],
                 "claim_amount_inr":[claim_amount],
                   "incident_type":[incident_type],
                  "incident_severity":[incident_severity],
                  "days_to_report":[days_to_report],
                   "police_report_filed":[police_report],
                    "witnesses":[witnesses],
                    "num_past_claims_3yrs":[past_claims],
                     "documents_complete":[documents_complete],
                     "claim_channel":[claim_channel],
                      "income_bracket":[income_bracket],
                       "claim_location":[claim_location]
                })

                processed = fraud_preprocessor.transform(input_df)

                prediction = fraud_model.predict(processed)

                if prediction[0] == "Y":
                 st.error("🚨 Fraudulent Claim Detected")
                else:
                 st.success("✅ Genuine Claim")
             



# ---------------- DAMAGE ----------------
    elif module == "🚗 Damage Detection":

     st.title("🚗 Vehicle  Damage Detection")

     st.write("upload a vehicle image for damage detection.")
     uploaded_file = st.file_uploader(
    "Choose Vehicle Image",
    type=["jpg", "jpeg", "png"]
      )
     if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
        image,
        caption="Uploaded Vehicle Image",
        use_container_width=True
       )
        detect_damage = st.button(
    "🚗 Detect Damage",
    use_container_width=True
     )
        if detect_damage:

         img = image.resize((224, 224))

         img_array = np.array(img)

         img_array = img_array / 255.0

         img_array = np.expand_dims(img_array, axis=0)

         prediction = damage_model.predict(img_array)
         st.write("Raw Prediction:", prediction)

         probability = prediction[0][0]

         if probability < 0.5:
           st.error("🚗💥 Prediction: Damaged Vehicle")
           confidence = (1 - probability) * 100
         else:
           st.success("🚗✅ Prediction: Whole Vehicle")
           confidence =  probability * 100
         st.metric(
           "Confidence Score",
            f"{confidence:.2f}%"
            )

           
# ---------------- SENTIMENT ----------------
    elif module == "😊 Review Sentiment":
     st.title("😊 Insurance Review Sentiment Analysis")

     st.write("Enter an insurance customer review below.")
     review = st.text_area(
        "Customer Review",
        height=180,
        placeholder="Type your insurance review here..."
      )
     st.write("")
     analyze = st.button(
        "😊 Analyze Sentiment",
        use_container_width=True
      )
     if analyze:

       sequence = tokenizer.texts_to_sequences([review])

       padded = pad_sequences(
        sequence,
        maxlen=100,
        padding="post"
      )

       st.success("Review processed successfully ✅")
       prediction = sentiment_model.predict(padded)

       predicted_class = np.argmax(prediction, axis=1)[0]

       confidence = np.max(prediction) * 100

       if predicted_class == 0:
         st.error("😡 Sentiment: Negative")

       elif predicted_class == 1:
         st.warning("😐 Sentiment: Neutral")

       else:
         st.success("😊 Sentiment: Positive")

       st.metric(
         "Confidence Score",
        f"{confidence:.2f}%"
)   



# ---------------- CHATBOT ----------------
    elif module == "🤖 InsureAI Chatbot":

       st.title("🤖 InsureAI Insurance Assistant")

       st.write("Ask any insurance-related question.")
       user_question = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Example: What is health insurance?"
    )
       ask = st.button(
        "🤖 Ask AI",
        use_container_width=True
    )
       if ask:
        if user_question.strip() == "":
            st.warning("Please enter a question.")

        else:
            prompt = f"""
    You are an Insurance AI Assistant.

    Answer only insurance-related questions.

    Keep the answers short, professional, and easy to understand.

    Question:
    {user_question}
    """ 
            try:
                 with st.spinner("🤖 AI is thinking..."):
                    response = chat_model.generate_content(prompt)

                 st.success("✅ Response Generated Successfully")

                 

                 st.success("🤖 AI Response")

                 st.write(response.text)
            except Exception as e:
                  st.error(f"Error: {e}")

               
           
