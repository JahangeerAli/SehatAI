```python
import os
import sys
import io
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.ocr_utils import extract_report_values
from utils.nlp_utils import parse_symptoms
from utils.fusion_utils import fuse_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SehatAI | Rural Health Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f6f9fc;
    }

    /* Main header */
    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #0f766e 0%,
            #0e7490 45%,
            #2563eb 100%
        );
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.15);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.95;
    }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 15px;
    }

    .section-header {
        font-size: 22px;
        font-weight: 750;
        margin: 8px 0 14px 0;
        color: #0f172a;
    }

    .risk-card {
        background: white;
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .risk-title {
        font-size: 15px;
        color: #64748b;
        font-weight: 600;
    }

    .risk-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 5px;
    }

    .disclaimer {
        padding: 13px 17px;
        border-radius: 12px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .chat-note {
        padding: 10px;
        border-radius: 10px;
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        color: #155e75;
        font-size: 13px;
    }

    /* Better buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    diabetes_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "diabetes_model.pkl"
        )
    )

    cardio_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "cardio_model.pkl"
        )
    )

    diabetes_features = joblib.load(
        os.path.join(
            MODELS_DIR,
            "diabetes_features.pkl"
        )
    )

    cardio_features = joblib.load(
        os.path.join(
            MODELS_DIR,
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

    st.error("❌ Could not load the trained ML models.")

    st.code(str(e))

    st.stop()


# ============================================================
# GROQ CONFIGURATION
# ============================================================

def get_groq_api_key():

    """
    Robustly search for GROQ_API_KEY.

    Works with:
    - Streamlit Cloud Secrets
    - environment variables
    """

    # --------------------------------------------------------
    # Streamlit Secrets
    # --------------------------------------------------------

    try:

        if "GROQ_API_KEY" in st.secrets:

            key = st.secrets["GROQ_API_KEY"]

            if key:
                return str(key).strip()

    except Exception:
        pass

    # --------------------------------------------------------
    # Environment variable
    # --------------------------------------------------------

    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    return None


def get_groq_client():

    api_key = get_groq_api_key()

    if not api_key:
        return None

    try:

        from groq import Groq

        client = Groq(
            api_key=api_key
        )

        return client

    except Exception:

        return None


# ============================================================
# GROQ MODEL
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# GENERAL GROQ CHAT FUNCTION
# ============================================================

def ask_groq(
    messages,
    max_tokens=700
):

    client = get_groq_client()

    if client is None:

        return None, (
            "Groq API is not configured. "
            "Please check GROQ_API_KEY in Streamlit Secrets."
        )

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=messages,

            temperature=0.2,

            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content

        return answer, None

    except Exception as e:

        return None, str(e)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        from pypdf import PdfReader

        pdf_bytes = uploaded_file.getvalue()

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    except Exception as e:

        return ""


# ============================================================
# PROCESS UPLOADED REPORT
# ============================================================

def process_report(uploaded_file):

    if uploaded_file is None:

        return {}, ""

    filename = uploaded_file.name.lower()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if filename.endswith(
        (".png", ".jpg", ".jpeg")
    ):

        try:

            image = Image.open(
                uploaded_file
            )

            temp_path = os.path.join(
                APP_DIR,
                "temp_report.png"
            )

            image.save(
                temp_path
            )

            values, raw_text = (
                extract_report_values(
                    temp_path
                )
            )

            return values, raw_text

        except Exception as e:

            st.warning(
                f"Could not process image report: {e}"
            )

            return {}, ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.endswith(".pdf"):

        text = extract_pdf_text(
            uploaded_file
        )

        if not text:

            st.warning(
                "The PDF does not contain selectable text. "
                "For scanned/image-only PDFs, upload the report "
                "as PNG/JPG or add PDF OCR support later."
            )

            return {}, ""

        # Use the same simple report patterns
        # on extracted PDF text.

        import re

        text_lower = text.lower()

        patterns = {

            "Glucose":
                r"glucose[:\s]+(\d+\.?\d*)",

            "trestbps":
                r"(?:blood pressure|bp)[:\s]+(\d+)",

            "chol":
                r"cholesterol[:\s]+(\d+\.?\d*)",

            "BMI":
                r"bmi[:\s]+(\d+\.?\d*)"
        }

        extracted = {}

        for key, pattern in patterns.items():

            match = re.search(
                pattern,
                text_lower
            )

            if match:

                extracted[key] = float(
                    match.group(1)
                )

        return extracted, text

    return {}, ""


# ============================================================
# AI PATIENT SUGGESTIONS
# ============================================================

def generate_patient_suggestions(
    age,
    sex,
    glucose,
    bmi,
    bp,
    cholesterol,
    diabetes_risk,
    cardio_risk,
    symptoms
):

    sex_name = (
        "Male"
        if sex == 1
        else "Female"
    )

    prompt = f"""
You are SehatAI, a cautious health education assistant.

You are NOT a doctor.

You must NOT:
- diagnose a disease
- prescribe medication
- recommend prescription doses
- claim certainty

You may provide general health education and recommend
appropriate professional medical evaluation.

Patient information:

Age: {age}
Sex: {sex_name}

Glucose: {glucose} mg/dL
BMI: {bmi}
Systolic BP: {bp} mmHg
Cholesterol: {cholesterol} mg/dL

Diabetes screening model:
{diabetes_risk * 100:.1f}%

Cardiovascular screening model:
{cardio_risk * 100:.1f}%

Symptoms:
{symptoms if symptoms else "No symptoms reported"}

Provide:

1. Simple explanation of the screening result.
2. 3-5 general healthy actions.
3. What the patient should discuss with a doctor.
4. Emergency warning signs.
5. A short reminder that this is NOT a diagnosis.

Use simple language.
Do not alarm the patient unnecessarily.
"""

    messages = [

        {
            "role": "system",
            "content": (
                "You are a safe health education assistant. "
                "Never diagnose or prescribe."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ]

    answer, error = ask_groq(
        messages,
        max_tokens=900
    )

    if error:

        return (
            "⚠️ AI suggestions are currently unavailable.\n\n"
            f"Technical message: {error}"
        )

    return answer


# ============================================================
# PATIENT CHATBOT
# ============================================================

def initialize_chat():

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = [

            {
                "role": "assistant",
                "content":
                (
                    "👋 Hello! I am the SehatAI health "
                    "education assistant.\n\n"
                    "Tell me what you are experiencing. "
                    "For example: **I have fever**.\n\n"
                    "I may ask follow-up questions such as "
                    "how long you have had the symptom."
                )
            }

        ]


def chatbot_response(user_message):

    context = """

You are SehatAI Patient Assistant.

You provide GENERAL HEALTH EDUCATION ONLY.

Never:
- diagnose
- prescribe medication
- recommend prescription doses
- replace a doctor

Conversation behavior:

If a patient gives a symptom, first ask useful
follow-up questions before giving detailed suggestions.

For example:

Patient:
"I have fever."

Ask:
1. How long have you had the fever?
2. What is your temperature, if measured?
3. Do you have cough, sore throat, breathing difficulty,
   vomiting, diarrhea, rash, severe headache, or pain?
4. Are you able to drink fluids?

After enough information is available:

- briefly summarize the symptoms
- give general self-care suggestions
- explain when professional medical evaluation is appropriate
- clearly list emergency warning signs

If the patient mentions emergency symptoms such as:
- severe difficulty breathing
- severe chest pain
- unconsciousness
- seizure
- severe confusion
- blue lips
- uncontrolled bleeding

tell them to seek urgent/emergency medical care.

Use simple English.
Be concise and friendly.
"""

    messages = [

        {
            "role": "system",
            "content": context
        }

    ]

    # Keep recent conversation
    messages.extend(
        st.session_state.chat_messages[-8:]
    )

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    answer, error = ask_groq(
        messages,
        max_tokens=600
    )

    if error:

        return (
            "⚠️ I cannot connect to the AI assistant right now.\n\n"
            f"Technical message: {error}"
        )

    return answer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🏥 SehatAI
        </div>

        <div class="hero-subtitle">
            Rural Health Risk & Triage Copilot
        </div>

        <div style="margin-top:10px;">
            AI-assisted screening • Patient education • Human review
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="disclaimer">
    ⚕️ <b>Important:</b> SehatAI is an AI-assisted
    screening and health-education prototype. It does not
    diagnose diseases or prescribe treatment. Results should
    be reviewed by a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR CHATBOT
# ============================================================

initialize_chat()

with st.sidebar:

    st.header("🤖 Patient Assistant")

    st.markdown(
        """
        <div class="chat-note">
        Describe your symptoms in your own words.
        The assistant may ask follow-up questions before
        giving general suggestions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Display messages

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    user_chat = st.chat_input(
        "Example: I have fever..."
    )

    if user_chat:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_chat
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_chat
            )

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer = chatbot_response(
                    user_chat
                )

            st.markdown(
                answer
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_messages = []

        st.rerun()

    st.caption(
        f"Model: {GROQ_MODEL}"
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩺 Patient Screening",
        "📄 Report Analysis",
        "ℹ️ How SehatAI Works"
    ]
)


# ============================================================
# TAB 1 — SCREENING
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-header">'
        '👤 Patient Information'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        name = st.text_input(
            "Patient name",
            placeholder="Optional"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35
        )

        sex_label = st.selectbox(
            "Sex",
            [
                "Female",
                "Male"
            ]
        )

        sex = (
            1
            if sex_label == "Male"
            else 0
        )

    with c2:

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=500,
            value=100
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=80.0,
            value=22.0,
            step=0.1
        )

        bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=0,
            max_value=300,
            value=120
        )

    with c3:

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=600,
            value=180
        )

        max_hr = st.number_input(
            "Maximum heart rate",
            min_value=0,
            max_value=300,
            value=150
        )

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )


    st.markdown(
        '<div class="section-header">'
        '📝 Symptoms'
        '</div>',
        unsafe_allow_html=True
    )

    symptoms_text = st.text_area(
        "Describe your symptoms",
        placeholder=(
            "Example: I have fatigue and excessive thirst..."
        ),
        height=110
    )


    st.markdown(
        '<div class="section-header">'
        '📎 Medical / Lab Report'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_report = st.file_uploader(
        "Upload PNG, JPG, JPEG or PDF",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf"
        ],
        accept_multiple_files=False,
        help=(
            "Upload an image or text-based PDF laboratory report."
        )
    )


    analyze = st.button(
        "🔍 Analyze My Health Risk",
        type="primary",
        use_container_width=True
    )


    if analyze:

        with st.spinner(
            "Analyzing patient information..."
        ):

            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            ocr_values = {}
            raw_text = ""

            if uploaded_report:

                ocr_values, raw_text = (
                    process_report(
                        uploaded_report
                    )
                )


            # ------------------------------------------------
            # SYMPTOMS
            # ------------------------------------------------

            symptom_adjustments = (
                parse_symptoms(
                    symptoms_text
                )
            )


            # ------------------------------------------------
            # DIABETES
            # ------------------------------------------------

            diabetes_input = {

                "Pregnancies":
                    pregnancies,

                "Glucose":
                    glucose,

                "BloodPressure":
                    bp,

                "SkinThickness":
                    20,

                "Insulin":
                    80,

                "BMI":
                    bmi,

                "DiabetesPedigreeFunction":
                    0.5,

                "Age":
                    age
            }


            diabetes_input = fuse_features(

                diabetes_input,

                symptom_adjustments,

                ocr_values
            )


            d_row = pd.DataFrame(
                [diabetes_input]
            )[diabetes_features]


            diabetes_risk = float(

                diabetes_model
                .predict_proba(d_row)[0][1]

            )


            # ------------------------------------------------
            # CARDIO
            # ------------------------------------------------

            cardio_input = {

                "age":
                    age,

                "sex":
                    sex,

                "cp":
                    0,

                "trestbps":
                    bp,

                "chol":
                    cholesterol,

                "fbs":
                    1
                    if glucose > 120
                    else 0,

                "restecg":
                    0,

                "thalach":
                    max_hr,

                "exang":
                    0,

                "oldpeak":
                    1.0,

                "slope":
                    1,

                "ca":
                    0,

                "thal":
                    2
            }


            cardio_input = fuse_features(

                cardio_input,

                symptom_adjustments,

                ocr_values
            )


            c_row = pd.DataFrame(
                [cardio_input]
            )[cardio_features]


            cardio_risk = float(

                cardio_model
                .predict_proba(c_row)[0][1]

            )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-header">'
            '📊 Screening Results'
            '</div>',
            unsafe_allow_html=True
        )


        overall = max(
            diabetes_risk,
            cardio_risk
        )


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

            st.markdown(
                f"""
                <div class="risk-card">

                <div class="risk-title">
                Diabetes Risk
                </div>

                <div class="risk-value">
                {diabetes_risk * 100:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with r2:

            st.markdown(
                f"""
                <div class="risk-card">

                <div class="risk-title">
                Cardiovascular Risk
                </div>

                <div class="risk-value">
                {cardio_risk * 100:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with r3:

            st.markdown(
                f"""
                <div class="risk-card">

                <div class="risk-title">
                Overall Screening
                </div>

                <div class="risk-value">
                {icon} {level}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")


        if overall >= 0.60:

            st.error(
                "🔴 Higher screening risk detected. "
                "Please arrange professional medical evaluation. "
                "If you have severe or emergency symptoms, seek "
                "urgent medical care."
            )

        elif overall >= 0.30:

            st.warning(
                "🟡 Moderate screening risk. "
                "Consider discussing these results with a "
                "healthcare professional."
            )

        else:

            st.success(
                "🟢 Lower model-estimated risk. "
                "Continue healthy habits and routine healthcare."
            )


        # ----------------------------------------------------
        # REPORT VALUES
        # ----------------------------------------------------

        if ocr_values:

            with st.expander(
                "📄 View extracted report values"
            ):

                st.json(
                    ocr_values
                )


        if raw_text:

            with st.expander(
                "🔎 View report text"
            ):

                st.text(
                    raw_text[:10000]
                )


        # ----------------------------------------------------
        # AI SUGGESTIONS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 Personalized Health Guidance"
        )


        with st.spinner(
            "Preparing patient-friendly guidance..."
        ):

            suggestions = (
                generate_patient_suggestions(

                    age=age,

                    sex=sex,

                    glucose=glucose,

                    bmi=bmi,

                    bp=bp,

                    cholesterol=cholesterol,

                    diabetes_risk=diabetes_risk,

                    cardio_risk=cardio_risk,

                    symptoms=symptoms_text
                )
            )


        st.markdown(
            suggestions
        )


        st.caption(
            "AI-generated information is educational only "
            "and should be reviewed by a healthcare professional."
        )


        # ----------------------------------------------------
        # SUPABASE
        # ----------------------------------------------------

        try:

            from utils.db_utils import save_record

            save_record(

                name,

                age,

                diabetes_risk,

                cardio_risk,

                f"{icon} {level}"
            )

            st.toast(
                "Record saved successfully ✅"
            )

        except Exception:

            # Supabase should never break screening.
            pass


# ============================================================
# TAB 2 — REPORT
# ============================================================

with tab2:

    st.subheader(
        "📄 Medical Report Analysis"
    )

    st.write(
        "Upload a PNG/JPG/JPEG image or a text-based PDF."
    )

    report = st.file_uploader(
        "Choose a report",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf"
        ],
        key="report_tab"
    )

    if report:

        values, text = process_report(
            report
        )

        if values:

            st.success(
                "Report values extracted."
            )

            st.json(
                values
            )

        if text:

            with st.expander(
                "View extracted text"
            ):

                st.text(
                    text[:15000]
                )


# ============================================================
# TAB 3 — HOW IT WORKS
# ============================================================

with tab3:

    st.subheader(
        "🔄 How SehatAI Works"
    )

    st.markdown(
        """
### 1️⃣ Patient Information

The patient enters:

- Age
- Sex
- Glucose
- BMI
- Blood pressure
- Cholesterol
- Maximum heart rate
- Symptoms

### 2️⃣ Medical Report

The patient can optionally upload:

- PNG
- JPG
- JPEG
- PDF

The application extracts available information.

### 3️⃣ Machine Learning

Two trained models estimate:

- Diabetes screening risk
- Cardiovascular screening risk

### 4️⃣ Risk Summary

The application displays:

- Diabetes risk
- Cardiovascular risk
- Overall screening level

### 5️⃣ Groq AI

The open-weight Groq-hosted model provides:

- Patient-friendly explanations
- General health suggestions
- Follow-up questions
- Warning signs

### 6️⃣ Human Review

A healthcare professional should review the information
before making a medical decision.

---

### 🚨 Emergency Warning

The AI assistant is NOT an emergency service.

For severe chest pain, severe breathing difficulty,
loss of consciousness, seizure, severe confusion,
blue lips, uncontrolled bleeding, or other serious
emergency symptoms, seek urgent medical care.
        """
    )
```
