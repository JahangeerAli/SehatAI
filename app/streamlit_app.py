import os
import sys
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(APP_DIR, "models")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.ocr_utils import extract_report_values
from utils.nlp_utils import parse_symptoms
from utils.fusion_utils import fuse_features

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="SehatAI | Rural Health Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: #f7f9fc;
    }
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .hero p {
        margin: .45rem 0 0 0;
        font-size: 1.02rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0,0,0,.05);
        text-align: center;
    }
    .risk-number {
        font-size: 2rem;
        font-weight: 700;
    }
    .small-note {
        color: #64748b;
        font-size: .9rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    diabetes_model = joblib.load(
        os.path.join(MODELS_DIR, "diabetes_model.pkl")
    )
    cardio_model = joblib.load(
        os.path.join(MODELS_DIR, "cardio_model.pkl")
    )
    diabetes_features = joblib.load(
        os.path.join(MODELS_DIR, "diabetes_features.pkl")
    )
    cardio_features = joblib.load(
        os.path.join(MODELS_DIR, "cardio_features.pkl")
    )
    return (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features,
    )


try:
    (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features,
    ) = load_models()
except Exception as e:
    st.error("Unable to load the trained models.")
    st.code(str(e))
    st.stop()

# ---------------------------------------------------------
# GROQ
# ---------------------------------------------------------
def get_groq_client():
    """Return a Groq client only when GROQ_API_KEY is configured."""
    try:
        from groq import Groq

        api_key = None

        # Streamlit Cloud / local secrets
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

        # Environment variable fallback
        api_key = api_key or os.getenv("GROQ_API_KEY")

        if not api_key:
            return None

        return Groq(api_key=api_key)

    except Exception:
        return None


def generate_patient_suggestions(
    age,
    sex,
    diabetes_risk,
    cardio_risk,
    symptoms,
    glucose,
    bmi,
    bp,
    cholesterol,
):
    client = get_groq_client()

    if client is None:
        return (
            "Groq suggestions are not configured yet. "
            "Add GROQ_API_KEY in Streamlit Secrets to enable AI-generated "
            "patient guidance."
        )

    sex_label = "Male" if sex == 1 else "Female"

    prompt = f"""
You are a health-education assistant for SehatAI, an AI-assisted screening
prototype. Do NOT diagnose diseases, prescribe medicines, or claim certainty.

Give simple, safe, patient-friendly guidance based ONLY on the information
below. Clearly separate:
1) What the screening result means
2) General healthy actions
3) When to contact a healthcare professional
4) Emergency warning signs

Patient:
Age: {age}
Sex: {sex_label}
Glucose: {glucose}
BMI: {bmi}
Systolic blood pressure: {bp}
Cholesterol: {cholesterol}
Diabetes model risk: {diabetes_risk * 100:.1f}%
Cardiovascular model risk: {cardio_risk * 100:.1f}%
Symptoms: {symptoms or "None reported"}

Use short bullet points. Mention that model output is a screening estimate,
not a diagnosis. If severe symptoms such as chest pain, severe breathing
difficulty, fainting, or other emergency symptoms are present, advise
urgent medical care.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You provide cautious health education only. "
                        "Never diagnose or prescribe."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI suggestions could not be generated right now: {e}"


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏥 SehatAI</h1>
        <p>Rural Health Risk & Triage Copilot</p>
        <p class="small-note" style="color:#e0f2fe;">
            AI-assisted screening • Explainable workflow • Human review required
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "⚕️ SehatAI is a screening/research prototype. It does not provide a "
    "medical diagnosis or prescription. Results should be reviewed by a "
    "qualified healthcare professional."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("🧭 About SehatAI")
    st.write(
        "Enter basic patient information, optional symptoms, and an optional "
        "lab-report image. The app estimates diabetes and cardiovascular risk."
    )

    st.divider()
    st.subheader("Workflow")
    st.write("1. Patient information")
    st.write("2. ML risk estimation")
    st.write("3. Optional OCR")
    st.write("4. AI health suggestions")
    st.write("5. Human/doctor review")

    st.divider()
    st.caption("Built with Python • Streamlit • Scikit-learn • Groq")

# ---------------------------------------------------------
# MAIN FORM
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🩺 Patient Screening", "ℹ️ How It Works"])

with tab1:
    st.markdown(
        '<div class="section-title">👤 Patient Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Patient name", placeholder="Optional")
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        sex_label = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex_label == "Male" else 0

    with col2:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=500, value=100
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=80.0, value=22.0, step=0.1
        )
        bp = st.number_input(
            "Systolic BP (mmHg)", min_value=0, max_value=300, value=120
        )

    with col3:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", min_value=0, max_value=600, value=180
        )
        max_hr = st.number_input(
            "Maximum heart rate", min_value=0, max_value=300, value=150
        )
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=0
        )

    st.markdown(
        '<div class="section-title">📝 Symptoms</div>',
        unsafe_allow_html=True,
    )

    symptoms_text = st.text_area(
        "Describe symptoms in your own words",
        placeholder="Example: fatigue, excessive thirst, chest pain...",
        height=100,
    )

    st.markdown(
        '<div class="section-title">📄 Optional Lab Report</div>',
        unsafe_allow_html=True,
    )

    uploaded_report = st.file_uploader(
        "Upload a JPG, JPEG or PNG lab report",
        type=["jpg", "jpeg", "png"],
    )

    analyze = st.button(
        "🔍 Analyze Health Risk",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        with st.spinner("Analyzing patient information..."):
            # OCR
            ocr_values = {}
            raw_text = ""

            if uploaded_report:
                try:
                    image = Image.open(uploaded_report)
                    temp_path = os.path.join(APP_DIR, "temp_report.png")
                    image.save(temp_path)
                    ocr_values, raw_text = extract_report_values(temp_path)
                except Exception as e:
                    st.warning(f"Could not process the uploaded report: {e}")

            # Symptom parsing
            symptom_adjustments = parse_symptoms(symptoms_text)

            # Diabetes
            diabetes_input = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": bmi,
                "DiabetesPedigreeFunction": 0.5,
                "Age": age,
            }

            diabetes_input = fuse_features(
                diabetes_input,
                symptom_adjustments,
                ocr_values,
            )

            d_row = pd.DataFrame([diabetes_input])[diabetes_features]
            diabetes_risk = float(
                diabetes_model.predict_proba(d_row)[0][1]
            )

            # Cardiovascular
            cardio_input = {
                "age": age,
                "sex": sex,
                "cp": 0,
                "trestbps": bp,
                "chol": cholesterol,
                "fbs": 1 if glucose > 120 else 0,
                "restecg": 0,
                "thalach": max_hr,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 2,
            }

            cardio_input = fuse_features(
                cardio_input,
                symptom_adjustments,
                ocr_values,
            )

            c_row = pd.DataFrame([cardio_input])[cardio_features]
            cardio_risk = float(
                cardio_model.predict_proba(c_row)[0][1]
            )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------
        st.divider()
        st.subheader("📊 Screening Results")

        overall = max(diabetes_risk, cardio_risk)

        if overall < 0.30:
            level = "Low"
            icon = "🟢"
        elif overall < 0.60:
            level = "Moderate"
            icon = "🟡"
        else:
            level = "High"
            icon = "🔴"

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Diabetes Risk",
                f"{diabetes_risk * 100:.1f}%",
            )

        with r2:
            st.metric(
                "Cardiovascular Risk",
                f"{cardio_risk * 100:.1f}%",
            )

        with r3:
            st.metric(
                "Overall Screening Level",
                f"{icon} {level}",
            )

        if overall >= 0.60:
            st.error(
                "High screening risk. Please arrange professional medical "
                "evaluation promptly. Emergency symptoms require urgent care."
            )
        elif overall >= 0.30:
            st.warning(
                "Moderate screening risk. Consider follow-up with a "
                "healthcare professional for proper assessment."
            )
        else:
            st.success(
                "Lower model-estimated risk. Continue healthy habits and "
                "routine healthcare as appropriate."
            )

        # OCR details
        if ocr_values:
            with st.expander("📄 View extracted report values"):
                st.json(ocr_values)

        if raw_text:
            with st.expander("🔎 View OCR text"):
                st.text(raw_text)

        # AI suggestions
        st.divider()
        st.subheader("🤖 Personalized Health Suggestions")

        with st.spinner("Generating patient-friendly suggestions..."):
            suggestions = generate_patient_suggestions(
                age=age,
                sex=sex,
                diabetes_risk=diabetes_risk,
                cardio_risk=cardio_risk,
                symptoms=symptoms_text,
                glucose=glucose,
                bmi=bmi,
                bp=bp,
                cholesterol=cholesterol,
            )

        st.markdown(suggestions)

        st.caption(
            "AI suggestions are educational guidance only and should be "
            "reviewed by a healthcare professional."
        )

        # Optional Supabase save
        try:
            from utils.db_utils import save_record

            save_record(
                name,
                age,
                diabetes_risk,
                cardio_risk,
                f"{icon} {level}",
            )
            st.toast("Record saved successfully ✅")
        except Exception:
            # Do not break the patient-facing app if Supabase is not configured.
            pass


with tab2:
    st.subheader("🔄 How SehatAI Works")

    st.markdown(
        """
        **1. Patient Input**  
        Basic information, vitals, symptoms and optional lab-report image.

        **2. OCR + Symptom Processing**  
        OCR extracts selected values from an uploaded report and the symptom
        module detects simple symptom keywords.

        **3. Machine Learning**  
        The trained diabetes and cardiovascular models estimate screening risk.

        **4. Risk Summary**  
        The app displays separate model risks and an overall screening level.

        **5. Groq AI Suggestions**  
        When `GROQ_API_KEY` is configured, Groq generates simple educational
        suggestions based on the screening information.

        **6. Human Review**  
        A healthcare professional should review the result before any medical
        decision.
        """
    )

    st.info(
        "🚨 Emergency symptoms such as severe chest pain, severe difficulty "
        "breathing, fainting, or sudden severe deterioration require urgent "
        "medical attention rather than waiting for an AI result."
    )
