import streamlit as st
import joblib
import pandas as pd
import shap
from PIL import Image
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.ocr_utils import extract_report_values
from utils.nlp_utils import parse_symptoms
from utils.fusion_utils import fuse_features

st.set_page_config(page_title="SehatAI", page_icon="🏥", layout="wide")

# ---- Load models ----
diabetes_model = joblib.load("models/diabetes_model.pkl")
cardio_model = joblib.load("models/cardio_model.pkl")
diabetes_features = joblib.load("models/diabetes_features.pkl")
cardio_features = joblib.load("models/cardio_features.pkl")

st.title("🏥 SehatAI — Rural Health Risk & Triage Copilot")

tab1, tab2 = st.tabs(["🩺 Patient Screening", "👨‍⚕️ Doctor Dashboard"])

with tab1:
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120, 35)
        glucose = st.number_input("Glucose", 0, 300, 100)
        bmi = st.number_input("BMI", 0.0, 60.0, 22.0)
        bp = st.number_input("Blood Pressure (systolic)", 0, 250, 120)
    with col2:
        cholesterol = st.number_input("Cholesterol", 0, 400, 180)
        max_hr = st.number_input("Max Heart Rate", 0, 250, 150)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)

    symptoms_text = st.text_area("Describe symptoms (in your own words)")
    uploaded_report = st.file_uploader("Upload lab report (image)", type=["jpg", "png", "jpeg"])

    if st.button("🔍 Analyze Risk"):
        ocr_values = {}
        if uploaded_report:
            img_path = "temp_report.png"
            Image.open(uploaded_report).save(img_path)
            ocr_values, raw_text = extract_report_values(img_path)
            st.info(f"OCR extracted: {ocr_values}")

        symptom_adjustments = parse_symptoms(symptoms_text)

        # ---- Diabetes prediction ----
        diabetes_input = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": 20, "Insulin": 80, "BMI": bmi,
            "DiabetesPedigreeFunction": 0.5, "Age": age
        }
        diabetes_input = fuse_features(diabetes_input, symptom_adjustments, ocr_values)
        d_row = pd.DataFrame([diabetes_input])[diabetes_features]
        diabetes_risk = diabetes_model.predict_proba(d_row)[0][1]

        # ---- Cardio prediction ----
        cardio_input = {
            "age": age, "sex": 1, "cp": 0, "trestbps": bp, "chol": cholesterol,
            "fbs": 1 if glucose > 120 else 0, "restecg": 0, "thalach": max_hr,
            "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
        }
        cardio_input = fuse_features(cardio_input, symptom_adjustments, ocr_values)
        c_row = pd.DataFrame([cardio_input])[cardio_features]
        cardio_risk = cardio_model.predict_proba(c_row)[0][1]

        st.divider()
        col3, col4 = st.columns(2)
        col3.metric("Diabetes Risk", f"{diabetes_risk*100:.1f}%")
        col4.metric("Cardiovascular Risk", f"{cardio_risk*100:.1f}%")

        overall = max(diabetes_risk, cardio_risk)
        level = "🟢 Low" if overall < 0.3 else "🟡 Medium" if overall < 0.6 else "🔴 High"
        st.subheader(f"Overall Risk Level: {level}")

        if overall >= 0.6:
            st.error("⚠️ Recommend immediate referral to nearest doctor/clinic.")
        elif overall >= 0.3:
            st.warning("Recommend follow-up checkup within 1-2 weeks.")
        else:
            st.success("Low risk — maintain healthy lifestyle, routine checkup advised.")

        # Save to Supabase
        try:
            from utils.db_utils import save_record
            save_record(name, age, diabetes_risk, cardio_risk, level)
            st.toast("Record saved ✅")
        except Exception as e:
            st.warning(f"Could not save record: {e}")

with tab2:
    st.subheader("Patient History (Doctor View)")
    try:
        from utils.db_utils import fetch_records
        records = fetch_records()
        st.dataframe(pd.DataFrame(records))
    except Exception as e:
        st.warning(f"Could not load records: {e}")
