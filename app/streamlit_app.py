import os
import sys
import io
import tempfile
import html

import joblib
import pandas as pd
import streamlit as st
from PIL import Image

# =========================================================
# OPTIONAL IMPORTS
# =========================================================

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from utils.ocr_utils import extract_report_values
except Exception:
    extract_report_values = None

try:
    from utils.nlp_utils import parse_symptoms
except Exception:
    parse_symptoms = None

try:
    from utils.db_utils import save_record
except Exception:
    save_record = None


# =========================================================
# PATHS
# =========================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="SehatAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f5f8fc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    /* ================= HERO ================= */

    .hero-box {
        background: linear-gradient(
            120deg,
            #087f8c,
            #0b9aaa,
            #2563eb
        );
        border-radius: 20px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 19px;
        font-weight: 600;
        margin-top: 7px;
    }

    .hero-description {
        font-size: 13px;
        margin-top: 8px;
        opacity: 0.95;
    }

    /* ================= SECTION ================= */

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    /* ================= CARDS ================= */

    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .card-title {
        font-size: 17px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .card-text {
        color: #475569;
        font-size: 14px;
        line-height: 1.65;
    }

    /* ================= RISK CARDS ================= */

    .risk-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        min-height: 145px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    .risk-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
    }

    .risk-number {
        font-size: 32px;
        font-weight: 800;
        color: #0f766e;
        margin-top: 8px;
    }

    .risk-level {
        font-size: 24px;
        font-weight: 800;
        color: #2563eb;
        margin-top: 12px;
    }

    /* ================= NOTICE ================= */

    .notice {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 15px 18px;
        border-radius: 10px;
        color: #1e3a8a;
        margin: 15px 0;
        line-height: 1.6;
    }

    .warning {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 16px 18px;
        border-radius: 10px;
        color: #7c2d12;
        margin: 15px 0;
        line-height: 1.6;
    }

    .emergency {
        background: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 16px 18px;
        border-radius: 10px;
        color: #7f1d1d;
        margin: 15px 0;
        line-height: 1.6;
    }

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    .chat-header {
        background: linear-gradient(
            135deg,
            #0f766e,
            #2563eb
        );
        color: white;
        padding: 15px;
        border-radius: 14px;
        margin-bottom: 12px;
    }

    .chat-header-title {
        font-size: 19px;
        font-weight: 750;
    }

    .chat-header-text {
        font-size: 12px;
        margin-top: 5px;
        opacity: 0.92;
        line-height: 1.5;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# IMPORTANT: One HTML block prevents raw HTML issue
# =========================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🏥 SehatAI</div>
        <div class="hero-subtitle">Rural Health Risk &amp; Triage Copilot</div>
        <div class="hero-description">
            AI-assisted screening &nbsp;•&nbsp;
            Patient education &nbsp;•&nbsp;
            Human review
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LANGUAGE
# =========================================================

language_col1, language_col2 = st.columns([3, 1])

with language_col2:

    language = st.selectbox(
        "🌐 Language",
        [
            "English",
            "اردو",
            "Roman Urdu"
        ]
    )


# =========================================================
# LANGUAGE INSTRUCTIONS
# =========================================================

def language_instruction():

    if language == "اردو":
        return """
Respond in simple Urdu.
Use easy words that ordinary patients can understand.
Avoid complicated medical terminology.
"""

    if language == "Roman Urdu":
        return """
Respond in simple Roman Urdu.
Use easy everyday language.
Avoid complicated medical terminology.
"""

    return """
Respond in simple English.
Use short, clear sentences.
Avoid complicated medical terminology.
"""


# =========================================================
# GROQ API KEY
# =========================================================

def get_groq_api_key():

    try:

        if "GROQ_API_KEY" in st.secrets:

            key = st.secrets["GROQ_API_KEY"]

            if key:
                return str(key).strip()

    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    return None


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    key = get_groq_api_key()

    if not key or Groq is None:
        return None

    return Groq(api_key=key)


# =========================================================
# GROQ CHAT
# =========================================================

def ask_groq(messages):

    client = get_groq_client()

    if client is None:

        return (
            "AI assistant is not configured yet. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.25,
            max_completion_tokens=800,
            include_reasoning=False
        )

        answer = response.choices[0].message.content

        if answer:
            return answer.strip()

        return "I could not generate a response right now."

    except Exception as e:

        error = str(e).lower()

        if "401" in error:

            return (
                "The AI service key is invalid or expired. "
                "Please check GROQ_API_KEY in Streamlit Secrets."
            )

        if "429" in error:

            return (
                "The AI service is temporarily busy or rate-limited. "
                "Please try again after a short while."
            )

        if "tool_use_failed" in error:

            return (
                "The AI assistant encountered a temporary "
                "conversation error. Please send your message again."
            )

        return (
            "I cannot connect to the AI assistant right now. "
            "Please try again in a moment."
        )


# =========================================================
# HEALTH CHATBOT SYSTEM PROMPT
# =========================================================

def chatbot_system_prompt():

    return f"""
You are SehatAI, a patient education and triage assistant.

{language_instruction()}

Your role:
- Understand symptoms.
- Ask appropriate follow-up questions.
- Identify possible emergency warning signs.
- Give general health education.
- Suggest reasonable next steps.

You are NOT a doctor.

Never:
- Diagnose with certainty.
- Prescribe medicine.
- Give medicine dosage.
- Tell the patient to stop prescribed medicine.
- Claim that an ML result proves a disease.

IMPORTANT CONVERSATION RULE:

When a patient gives only a symptom without enough information,
DO NOT immediately give a long health lecture.

Instead, ask 2 or 3 short follow-up questions.

For example:

Patient:
"I have fever."

Your response should be similar to:

"I'm sorry you're not feeling well.
How long have you had the fever?
Have you measured your temperature?
Do you have any other symptoms such as cough,
sore throat, vomiting, diarrhea, rash, severe headache,
or breathing difficulty?"

Then WAIT for the patient's answer.

After the patient provides enough information,
give organized guidance:

1. What it may mean generally
2. What they can do now
3. What to monitor
4. When to contact a doctor
5. Emergency warning signs

Do not overwhelm the patient.

If the patient reports:
- severe difficulty breathing
- severe chest pain
- fainting
- confusion
- seizure
- blue lips
- severe bleeding
- sudden weakness on one side
- severe allergic reaction

recommend urgent/emergency medical care immediately.

For fever, ask about:
- duration
- measured temperature
- age
- other symptoms
- hydration
- whether symptoms are improving or worsening

Use a warm, respectful and supportive tone.
"""


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    if PdfReader is None:
        return ""

    try:

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception:

        return ""


# =========================================================
# REPORT PROCESSING
# =========================================================

def process_report(uploaded_file):

    if uploaded_file is None:
        return {}, ""

    filename = uploaded_file.name.lower()

    # ---------------- PDF ----------------

    if filename.endswith(".pdf"):

        text = extract_pdf_text(
            uploaded_file
        )

        return {}, text

    # ---------------- IMAGE ----------------

    try:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        if extract_report_values is None:
            return {}, ""

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp:

            image.save(temp.name)

            values, text = extract_report_values(
                temp.name
            )

        return values, text

    except Exception:

        return {}, ""


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    diabetes_model = joblib.load(
        os.path.join(
            APP_DIR,
            "models",
            "diabetes_model.pkl"
        )
    )

    cardio_model = joblib.load(
        os.path.join(
            APP_DIR,
            "models",
            "cardio_model.pkl"
        )
    )

    diabetes_features = joblib.load(
        os.path.join(
            APP_DIR,
            "models",
            "diabetes_features.pkl"
        )
    )

    cardio_features = joblib.load(
        os.path.join(
            APP_DIR,
            "models",
            "cardio_features.pkl"
        )
    )

    return (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features
    )


try:

    (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features
    ) = load_models()

except Exception as e:

    st.error(
        "⚠️ Model files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# SIDEBAR CHATBOT
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-header-title">
                🤖 SehatAI Health Assistant
            </div>
            <div class="chat-header-text">
                Tell me your symptoms. I will ask
                follow-up questions before giving
                general guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    chatbot_language = st.selectbox(
        "Chat language",
        [
            "English",
            "اردو",
            "Roman Urdu"
        ],
        key="chat_language"
    )

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋\n\n"
                    "I'm SehatAI. Tell me what you are "
                    "experiencing.\n\n"
                    "For example: **I have fever.**"
                )
            }
        ]

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    user_message = st.chat_input(
        "Describe your symptoms..."
    )

    if user_message:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_message
            )

        # Chat language override
        if chatbot_language == "اردو":

            chat_language_instruction = """
Respond in simple Urdu.
"""

        elif chatbot_language == "Roman Urdu":

            chat_language_instruction = """
Respond in simple Roman Urdu.
"""

        else:

            chat_language_instruction = """
Respond in simple English.
"""

        system_prompt = chatbot_system_prompt()

        system_prompt += chat_language_instruction

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Keep conversation manageable
        for message in (
            st.session_state.chat_messages[-12:]
        ):

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "SehatAI is thinking..."
            ):

                answer = ask_groq(
                    messages
                )

            st.markdown(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Chat cleared. 👋\n\n"
                    "Tell me your symptoms whenever "
                    "you are ready."
                )
            }
        ]

        st.rerun()


# =========================================================
# MAIN TABS
# =========================================================

tab_screening, tab_report, tab_about = st.tabs(
    [
        "🩺 Patient Screening",
        "📄 Report Analysis",
        "ℹ️ About SehatAI"
    ]
)


# =========================================================
# PATIENT SCREENING
# =========================================================

with tab_screening:

    st.markdown(
        '<div class="section-title">👤 Patient Information</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

    with p2:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    with p3:

        sex_label = st.selectbox(
            "Sex",
            [
                "Male",
                "Female"
            ]
        )

    sex = 1 if sex_label == "Male" else 0

    st.markdown(
        '<div class="section-title">🩸 Diabetes Screening</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

    with d2:

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=500.0,
            value=100.0
        )

    with d3:

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=300.0,
            value=70.0
        )

    with d4:

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=200.0,
            value=20.0
        )

    d5, d6, d7, d8 = st.columns(4)

    with d5:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

    with d6:

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

    with d7:

        pedigree = st.number_input(
            "Diabetes Pedigree",
            min_value=0.0,
            max_value=5.0,
            value=0.47
        )

    with d8:

        diabetes_age = st.number_input(
            "Diabetes Age",
            min_value=1,
            max_value=120,
            value=int(age)
        )

    st.markdown(
        '<div class="section-title">❤️ Cardiovascular Screening</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

    with c2:

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50.0,
            max_value=300.0,
            value=120.0
        )

    with c3:

        chol = st.number_input(
            "Cholesterol",
            min_value=50.0,
            max_value=700.0,
            value=200.0
        )

    with c4:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1]
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with c6:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50.0,
            max_value=250.0,
            value=150.0
        )

    with c7:

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1]
        )

    with c8:

        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0
        )

    c9, c10, c11 = st.columns(3)

    with c9:

        slope = st.selectbox(
            "ST Slope",
            [0, 1, 2]
        )

    with c10:

        ca = st.selectbox(
            "Major Vessels (CA)",
            [0, 1, 2, 3, 4]
        )

    with c11:

        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3]
        )

    # =====================================================
    # SYMPTOMS
    # =====================================================

    st.markdown(
        '<div class="section-title">📝 Symptoms</div>',
        unsafe_allow_html=True
    )

    symptoms = st.text_area(
        "Describe symptoms",
        placeholder=(
            "Example: I have fever since yesterday, "
            "temperature is 38.5°C, and I have a sore throat."
        ),
        height=100
    )

    # =====================================================
    # REPORT
    # =====================================================

    st.markdown(
        '<div class="section-title">📄 Medical Report</div>',
        unsafe_allow_html=True
    )

    report = st.file_uploader(
        "Upload PNG, JPG, JPEG or PDF",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf"
        ],
        help="Maximum file size depends on your Streamlit configuration."
    )

    # =====================================================
    # SCREEN BUTTON
    # =====================================================

    run_screening = st.button(
        "🔎 Run Health Screening",
        type="primary",
        use_container_width=True
    )

    if run_screening:

        # -------------------------------------------------
        # INPUT DATA
        # -------------------------------------------------

        diabetes_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": diabetes_age
        }

        cardio_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------

        ocr_values = {}
        report_text = ""

        if report:

            with st.spinner(
                "📄 Reading medical report..."
            ):

                ocr_values, report_text = process_report(
                    report
                )

        # -------------------------------------------------
        # SYMPTOMS
        # -------------------------------------------------

        symptom_adjustments = {}

        if symptoms and parse_symptoms:

            try:

                symptom_adjustments = parse_symptoms(
                    symptoms
                )

            except Exception:

                symptom_adjustments = {}

        # -------------------------------------------------
        # OPTIONAL OCR VALUES
        # -------------------------------------------------

        # Only replace values that actually exist
        # in the corresponding model inputs.

        for key, value in ocr_values.items():

            if key in diabetes_data:

                diabetes_data[key] = value

            if key in cardio_data:

                cardio_data[key] = value

        # -------------------------------------------------
        # FEATURE ALIGNMENT
        # -------------------------------------------------

        diabetes_input = {}

        for feature in diabetes_features:

            diabetes_input[feature] = diabetes_data.get(
                feature,
                0
            )

        cardio_input = {}

        for feature in cardio_features:

            cardio_input[feature] = cardio_data.get(
                feature,
                0
            )

        diabetes_df = pd.DataFrame(
            [diabetes_input]
        )

        cardio_df = pd.DataFrame(
            [cardio_input]
        )

        # -------------------------------------------------
        # DIABETES PREDICTION
        # -------------------------------------------------

        try:

            if hasattr(
                diabetes_model,
                "predict_proba"
            ):

                diabetes_risk = float(
                    diabetes_model
                    .predict_proba(
                        diabetes_df
                    )[0][1]
                )

            else:

                diabetes_risk = float(
                    diabetes_model
                    .predict(
                        diabetes_df
                    )[0]
                )

        except Exception as e:

            st.error(
                "Diabetes model prediction failed."
            )

            st.code(str(e))

            diabetes_risk = 0.0

        # -------------------------------------------------
        # CARDIO PREDICTION
        # -------------------------------------------------

        try:

            if hasattr(
                cardio_model,
                "predict_proba"
            ):

                cardio_risk = float(
                    cardio_model
                    .predict_proba(
                        cardio_df
                    )[0][1]
                )

            else:

                cardio_risk = float(
                    cardio_model
                    .predict(
                        cardio_df
                    )[0]
                )

        except Exception as e:

            st.error(
                "Cardiovascular model prediction failed."
            )

            st.code(str(e))

            cardio_risk = 0.0

        # -------------------------------------------------
        # OVERALL
        # -------------------------------------------------

        overall = (
            diabetes_risk +
            cardio_risk
        ) / 2

        if overall >= 0.70:

            overall_level = "Higher"

        elif overall >= 0.40:

            overall_level = "Moderate"

        else:

            overall_level = "Lower"

        # =================================================
        # RESULTS
        # =================================================

        st.markdown(
            '<div class="section-title">📊 Your Screening Summary</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-label">
                        🩸 Diabetes Screening
                    </div>
                    <div class="risk-number">
                        {diabetes_risk * 100:.1f}%
                    </div>
                    <div class="risk-label">
                        Estimated screening risk
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-label">
                        ❤️ Cardiovascular Screening
                    </div>
                    <div class="risk-number">
                        {cardio_risk * 100:.1f}%
                    </div>
                    <div class="risk-label">
                        Estimated screening risk
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r3:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-label">
                        📌 Overall Screening Level
                    </div>
                    <div class="risk-level">
                        {overall_level}
                    </div>
                    <div class="risk-label">
                        Screening estimate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # WHAT RESULT MEANS
        # =================================================

        st.markdown(
            '<div class="section-title">💡 What These Results Mean</div>',
            unsafe_allow_html=True
        )

        if language == "اردو":

            explanation = f"""
            <b>ذیابیطس اسکریننگ:</b> {diabetes_risk * 100:.1f}%<br>
            <b>دل اور خون کی نالیوں کی اسکریننگ:</b> {cardio_risk * 100:.1f}%<br>
            <b>مجموعی سطح:</b> {overall_level}<br><br>

            یہ نتائج صرف کمپیوٹر ماڈل کی اسکریننگ کا اندازہ ہیں۔
            یہ بیماری کی تشخیص نہیں ہیں۔
            حتمی طبی فیصلہ ڈاکٹر یا qualified healthcare professional کرے۔
            """

        elif language == "Roman Urdu":

            explanation = f"""
            <b>Diabetes screening:</b> {diabetes_risk * 100:.1f}%<br>
            <b>Heart/cardiovascular screening:</b> {cardio_risk * 100:.1f}%<br>
            <b>Overall level:</b> {overall_level}<br><br>

            Ye results sirf computer model ki screening estimate hain.
            Ye kisi disease ki diagnosis nahi hain.
            Final medical decision qualified healthcare professional kare.
            """

        else:

            explanation = f"""
            <b>Diabetes screening:</b> {diabetes_risk * 100:.1f}%<br>
            <b>Cardiovascular screening:</b> {cardio_risk * 100:.1f}%<br>
            <b>Overall level:</b> {overall_level}<br><br>

            These results are estimates from a computer screening model.
            They are not a medical diagnosis.
            A qualified healthcare professional should review important cases.
            """

        st.markdown(
            f"""
            <div class="card">
                <div class="card-text">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # IMPORTANT CONTEXT
        # =================================================

        st.markdown(
            """
            <div class="notice">
                <strong>ℹ️ Important:</strong>
                A screening percentage is not the same thing as
                a confirmed probability of developing a disease.
                The result depends on the training data and model.
                Your healthcare professional should interpret it
                together with your medical history and examination.
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # SYMPTOMS
        # =================================================

        if symptoms:

            st.markdown(
                '<div class="section-title">📝 Symptoms You Reported</div>',
                unsafe_allow_html=True
            )

            safe_symptoms = html.escape(
                symptoms
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-text">
                        {safe_symptoms}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # REPORT RESULTS
        # =================================================

        if ocr_values:

            st.markdown(
                '<div class="section-title">📄 Report Values Detected</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                pd.DataFrame(
                    list(
                        ocr_values.items()
                    ),
                    columns=[
                        "Parameter",
                        "Value"
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )

        if report_text:

            with st.expander(
                "📄 View extracted report text"
            ):

                st.write(
                    report_text
                )

        # =================================================
        # AI GUIDANCE
        # =================================================

        st.markdown(
            '<div class="section-title">🤖 SehatAI Guidance</div>',
            unsafe_allow_html=True
        )

        guidance_prompt = f"""
Create a short, well-organized patient guidance report.

Patient:
- Age: {age}
- Sex: {sex_label}

Screening:
- Diabetes: {diabetes_risk * 100:.1f}%
- Cardiovascular: {cardio_risk * 100:.1f}%
- Overall level: {overall_level}

Symptoms:
{symptoms if symptoms else "No symptoms reported."}

Report text:
{report_text[:2500] if report_text else "No report text available."}

Write using these headings:

### 1. Screening Summary
Explain the results simply.

### 2. What This Means
Explain that the numbers are model-based screening estimates,
not a diagnosis.

### 3. Healthy Next Steps
Give practical general health advice.

### 4. About the Report
Mention important extracted information if available.

### 5. When to Contact a Doctor
Give reasonable non-emergency reasons.

### 6. Emergency Warning Signs
Mention serious symptoms that require urgent medical attention.

Do not diagnose.
Do not prescribe medicine.
Do not give medicine dosage.
Do not say the patient definitely has or does not have a disease.

{language_instruction()}
"""

        with st.spinner(
            "🤖 Preparing your health guidance..."
        ):

            guidance = ask_groq(
                [
                    {
                        "role": "system",
                        "content": """
You are a careful health education assistant.
Your output must be organized, concise and patient-friendly.
Never diagnose or prescribe.
"""
                    },
                    {
                        "role": "user",
                        "content": guidance_prompt
                    }
                ]
            )

        st.markdown(
            f"""
            <div class="card">
                <div class="card-text">
                    {guidance.replace(chr(10), "<br>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # EMERGENCY NOTICE
        # =================================================

        st.markdown(
            """
            <div class="emergency">
                <strong>🚨 Emergency warning</strong><br><br>

                If you or the patient has severe difficulty breathing,
                severe chest pain, fainting, confusion, seizure,
                blue lips, severe bleeding, sudden weakness on one side,
                or another serious emergency symptom, seek emergency
                medical care immediately.
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # SAVE
        # =================================================

        if save_record is not None:

            try:

                save_record(
                    patient_name or "Anonymous",
                    age,
                    diabetes_risk * 100,
                    cardio_risk * 100,
                    overall_level
                )

                st.success(
                    "✅ Screening record saved."
                )

            except Exception:

                st.info(
                    "Screening completed. "
                    "History storage is not currently configured."
                )


# =========================================================
# REPORT TAB
# =========================================================

with tab_report:

    st.markdown(
        '<div class="section-title">📄 Medical Report Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Upload a medical report
            </div>
            <div class="card-text">
                SehatAI can read text from PDF reports and
                use OCR for image reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    standalone_report = st.file_uploader(
        "Choose a report",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf"
        ],
        key="standalone_report"
    )

    if standalone_report:

        filename = standalone_report.name.lower()

        if filename.endswith(".pdf"):

            text = extract_pdf_text(
                standalone_report
            )

            if text:

                st.success(
                    "✅ Text extracted successfully."
                )

                with st.expander(
                    "View extracted text"
                ):

                    st.write(text)

            else:

                st.warning(
                    "This PDF does not contain selectable text. "
                    "It may be a scanned PDF."
                )

        else:

            try:

                image = Image.open(
                    io.BytesIO(
                        standalone_report.getvalue()
                    )
                )

                st.image(
                    image,
                    caption="Uploaded Medical Report",
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "Unable to display this image."
                )

            if extract_report_values:

                with st.spinner(
                    "🔍 Running OCR..."
                ):

                    values, text = process_report(
                        standalone_report
                    )

                if values:

                    st.success(
                        "Values detected."
                    )

                    st.dataframe(
                        pd.DataFrame(
                            list(values.items()),
                            columns=[
                                "Parameter",
                                "Value"
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                if text:

                    with st.expander(
                        "View OCR text"
                    ):

                        st.write(text)


# =========================================================
# ABOUT TAB
# =========================================================

with tab_about:

    st.markdown(
        '<div class="section-title">ℹ️ About SehatAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                🏥 What is SehatAI?
            </div>
            <div class="card-text">
                SehatAI is an AI-assisted health screening and
                patient education prototype designed for
                resource-constrained and rural healthcare workflows.
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                🧠 Machine Learning
            </div>
            <div class="card-text">
                The prototype uses trained machine-learning models
                for diabetes and cardiovascular screening.
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                📄 Report Processing
            </div>
            <div class="card-text">
                Text-based PDF reports and image reports can be
                processed to extract useful information.
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                🤖 AI Assistant
            </div>
            <div class="card-text">
                The SehatAI assistant can have a conversational
                symptom discussion, ask follow-up questions and
                provide general health education.
            </div>
        </div>

        <div class="warning">
            <strong>⚠️ Medical Safety Notice</strong><br><br>

            SehatAI is a research and decision-support prototype.
            It is not a doctor and does not provide a medical
            diagnosis or prescription.

            AI-generated information can be incomplete or incorrect.
            Important decisions should be made with a qualified
            healthcare professional.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🏥 <strong>SehatAI</strong> —
        Rural Health Risk &amp; Triage Copilot
        <br>
        AI-assisted screening • Patient education • Human review
    </div>
    """,
    unsafe_allow_html=True
)
