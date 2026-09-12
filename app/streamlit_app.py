import os
import sys
import io
import joblib
import pandas as pd
import streamlit as st

from PIL import Image

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

# ---------------------------------------------------------
# OPTIONAL IMPORTS
# ---------------------------------------------------------

try:
    from utils.ocr_utils import extract_report_values
except Exception:
    extract_report_values = None

try:
    from utils.nlp_utils import parse_symptoms
except Exception:
    parse_symptoms = None

try:
    from utils.fusion_utils import fuse_features
except Exception:
    fuse_features = None

try:
    from utils.db_utils import save_record
except Exception:
    save_record = None

# ---------------------------------------------------------
# GROQ
# ---------------------------------------------------------

try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SehatAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef6ff 50%,
            #f8fafc 100%
        );
    }

    /* Remove excessive top spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero */
    .hero {
        padding: 30px 35px;
        border-radius: 24px;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            #0f766e 0%,
            #0891b2 50%,
            #2563eb 100%
        );
        color: white;
        box-shadow: 0 12px 35px rgba(15, 118, 110, 0.20);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .hero-description {
        font-size: 15px;
        opacity: 0.95;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    .risk-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    }

    .risk-number {
        font-size: 34px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Disclaimer */
    .disclaimer {
        padding: 15px 18px;
        border-radius: 12px;
        background: #fff7ed;
        border-left: 5px solid #f97316;
        color: #7c2d12;
        margin-top: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    /* Chat */
    .chat-info {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏥 SehatAI</div>

        <div class="hero-subtitle">
            Rural Health Risk &amp; Triage Copilot
        </div>

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
# CONSTANTS
# =========================================================

GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    diabetes_model_path = os.path.join(
        APP_DIR,
        "models",
        "diabetes_model.pkl"
    )

    cardio_model_path = os.path.join(
        APP_DIR,
        "models",
        "cardio_model.pkl"
    )

    diabetes_features_path = os.path.join(
        APP_DIR,
        "models",
        "diabetes_features.pkl"
    )

    cardio_features_path = os.path.join(
        APP_DIR,
        "models",
        "cardio_features.pkl"
    )

    diabetes_model = joblib.load(diabetes_model_path)
    cardio_model = joblib.load(cardio_model_path)

    diabetes_features = joblib.load(diabetes_features_path)
    cardio_features = joblib.load(cardio_features_path)

    return (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features
    )


# =========================================================
# GROQ API KEY
# =========================================================

def get_groq_api_key():

    # Streamlit Secrets
    try:
        if "GROQ_API_KEY" in st.secrets:

            key = st.secrets["GROQ_API_KEY"]

            if key:
                return str(key).strip()

    except Exception:
        pass

    # Environment variable
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    return None


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    api_key = get_groq_api_key()

    if not api_key:
        return None

    if Groq is None:
        return None

    return Groq(api_key=api_key)


# =========================================================
# GROQ CHAT FUNCTION
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

            # Important:
            # DO NOT use tool_choice="none".
            # GPT-OSS can attempt tool calling and that can
            # produce the 400 error seen previously.
            #
            # We provide NO tools at all.

            temperature=0.3,
            max_completion_tokens=700,

            # We do not need model reasoning displayed.
            include_reasoning=False
        )

        content = response.choices[0].message.content

        if content:
            return content.strip()

        return (
            "I could not generate a response right now. "
            "Please try again."
        )

    except Exception as e:

        error_text = str(e)

        # Friendly handling for the previous tool-use error
        if "tool_use_failed" in error_text.lower():

            return (
                "The AI assistant temporarily tried to use an "
                "unsupported tool mode. Please send your message "
                "again. No patient information was lost."
            )

        if "401" in error_text:
            return (
                "The Groq API key appears to be invalid or expired. "
                "Please check GROQ_API_KEY in Streamlit Secrets."
            )

        if "429" in error_text:
            return (
                "The AI service is temporarily rate-limited. "
                "Please wait a moment and try again."
            )

        if "model" in error_text.lower() and "not found" in error_text.lower():
            return (
                "The selected Groq model is currently unavailable. "
                "Please check the Groq model configuration."
            )

        return (
            "I cannot connect to the AI assistant right now. "
            "Please try again in a moment."
        )


# =========================================================
# HEALTH SYSTEM PROMPT
# =========================================================

HEALTH_SYSTEM_PROMPT = """
You are SehatAI, a patient education and health triage assistant.

You are NOT a doctor and must not diagnose diseases or prescribe medicines.

Your job is to:
1. Understand the patient's symptoms.
2. Ask useful follow-up questions.
3. Identify possible urgency.
4. Provide general health education.
5. Suggest appropriate next steps.
6. Encourage professional medical evaluation when appropriate.

For symptoms, ask concise follow-up questions such as:
- How long have you had the symptom?
- How severe is it?
- What is the measured temperature or vital sign?
- Are there other symptoms?
- Can the person drink fluids?
- Has the symptom been getting better or worse?

For emergency warning signs such as:
- severe difficulty breathing
- severe chest pain
- fainting
- confusion
- blue lips
- severe bleeding
- seizure
- sudden weakness on one side
- severe allergic reaction

tell the user to seek emergency medical care immediately.

Do not claim certainty.

Do not prescribe medication or dosage.

Use simple language that a rural patient can understand.

Always clearly distinguish general educational guidance from medical diagnosis.
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

        text_parts = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                text_parts.append(text)

        return "\n".join(text_parts)

    except Exception:
        return ""


# =========================================================
# REPORT PROCESSING
# =========================================================

def process_report(uploaded_file):

    if uploaded_file is None:
        return {}, ""

    file_name = uploaded_file.name.lower()

    # PDF
    if file_name.endswith(".pdf"):

        text = extract_pdf_text(uploaded_file)

        return {}, text

    # IMAGE
    try:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        if extract_report_values is not None:

            # Save temporary image
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
            ) as temp_file:

                image.save(temp_file.name)

                values, text = extract_report_values(
                    temp_file.name
                )

            return values, text

        return {}, ""

    except Exception as e:

        st.warning(
            f"Could not process the report: {e}"
        )

        return {}, ""


# =========================================================
# SIDEBAR CHATBOT
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 SehatAI Assistant")

    st.markdown(
        """
        <div class="chat-info">
        Describe your symptoms in simple language.
        The assistant may ask follow-up questions before
        providing general guidance.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm SehatAI. "
                    "Tell me what you are experiencing. "
                    "For example: "
                    "\"I have had fever since yesterday.\""
                )
            }
        ]

    # Display chat history
    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    user_message = st.chat_input(
        "Describe your symptoms..."
    )

    if user_message:

        # Add user message
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        # Build messages for Groq
        groq_messages = [
            {
                "role": "system",
                "content": HEALTH_SYSTEM_PROMPT
            }
        ]

        # Keep last 10 messages
        recent_messages = (
            st.session_state.chat_messages[-10:]
        )

        for msg in recent_messages:

            groq_messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
            )

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = ask_groq(groq_messages)

            st.markdown(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.button("🗑️ Clear chat"):

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Chat cleared. Tell me your symptoms "
                    "whenever you are ready."
                )
            }
        ]

        st.rerun()


# =========================================================
# LOAD ML MODELS
# =========================================================

try:

    (
        diabetes_model,
        cardio_model,
        diabetes_features,
        cardio_features
    ) = load_models()

except Exception as e:

    st.error(
        "Model files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩺 Patient Screening",
        "📄 Report Analysis",
        "ℹ️ How It Works"
    ]
)


# =========================================================
# TAB 1 — PATIENT SCREENING
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter name"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    with col2:

        gender = st.selectbox(
            "Sex",
            [
                "Male",
                "Female"
            ]
        )

        # UCI Cleveland convention:
        # Male = 1
        # Female = 0
        sex = 1 if gender == "Male" else 0

    with col3:

        report = st.file_uploader(
            "Medical Report",
            type=[
                "png",
                "jpg",
                "jpeg",
                "pdf"
            ],
            help="Upload PNG, JPG, JPEG or PDF."
        )

    st.markdown(
        '<div class="section-title">Diabetes Screening</div>',
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

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0
        )

    d5, d6, d7, d8 = st.columns(4)

    with d5:

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=200.0,
            value=20.0
        )

    with d6:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

    with d7:

        diabetes_pedigree = st.number_input(
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
        '<div class="section-title">Cardiovascular Screening</div>',
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
            "Resting BP",
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

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50.0,
            max_value=250.0,
            value=150.0
        )

    c5, c6 = st.columns(2)

    with c5:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1]
        )

    with c6:

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    st.markdown(
        '<div class="section-title">Symptoms</div>',
        unsafe_allow_html=True
    )

    symptoms = st.text_area(
        "Describe symptoms",
        placeholder=(
            "Example: fever, fatigue, excessive thirst, "
            "chest pain, shortness of breath..."
        ),
        height=100
    )

    screen_button = st.button(
        "🔎 Run Health Screening",
        type="primary",
        use_container_width=True
    )

    if screen_button:

        # -----------------------------------------------
        # BASE FEATURE DATA
        # -----------------------------------------------

        diabetes_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
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
            "thalach": thalach
        }

        # -----------------------------------------------
        # REPORT EXTRACTION
        # -----------------------------------------------

        ocr_values = {}
        report_text = ""

        if report is not None:

            with st.spinner(
                "Analyzing uploaded report..."
            ):

                ocr_values, report_text = process_report(
                    report
                )

        # -----------------------------------------------
        # SYMPTOM PROCESSING
        # -----------------------------------------------

        symptom_adjustments = {}

        if symptoms and parse_symptoms is not None:

            try:

                symptom_adjustments = parse_symptoms(
                    symptoms
                )

            except Exception:

                symptom_adjustments = {}

        # -----------------------------------------------
        # OPTIONAL FEATURE FUSION
        # -----------------------------------------------

        if fuse_features is not None:

            try:

                diabetes_data = fuse_features(
                    diabetes_data,
                    symptom_adjustments,
                    ocr_values
                )

                cardio_data = fuse_features(
                    cardio_data,
                    symptom_adjustments,
                    ocr_values
                )

            except Exception:
                pass

        # -----------------------------------------------
        # ALIGN FEATURES TO TRAINING DATA
        # -----------------------------------------------

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

        # -----------------------------------------------
        # PREDICTIONS
        # -----------------------------------------------

        try:

            if hasattr(
                diabetes_model,
                "predict_proba"
            ):

                diabetes_probability = (
                    diabetes_model
                    .predict_proba(diabetes_df)[0][1]
                )

            else:

                diabetes_probability = float(
                    diabetes_model.predict(
                        diabetes_df
                    )[0]
                )

        except Exception as e:

            st.error(
                "Diabetes model prediction failed."
            )

            st.code(str(e))

            diabetes_probability = 0.0

        try:

            if hasattr(
                cardio_model,
                "predict_proba"
            ):

                cardio_probability = (
                    cardio_model
                    .predict_proba(cardio_df)[0][1]
                )

            else:

                cardio_probability = float(
                    cardio_model.predict(
                        cardio_df
                    )[0]
                )

        except Exception as e:

            st.error(
                "Cardiovascular model prediction failed."
            )

            st.code(str(e))

            cardio_probability = 0.0

        # -----------------------------------------------
        # RISK LEVEL
        # -----------------------------------------------

        average_risk = (
            diabetes_probability +
            cardio_probability
        ) / 2

        if average_risk >= 0.70:

            risk_level = "High"

        elif average_risk >= 0.40:

            risk_level = "Moderate"

        else:

            risk_level = "Lower"

        # -----------------------------------------------
        # RESULTS
        # -----------------------------------------------

        st.markdown(
            '<div class="section-title">Screening Results</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div>Diabetes Risk</div>
                    <div class="risk-number">
                        {diabetes_probability * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div>Cardiovascular Risk</div>
                    <div class="risk-number">
                        {cardio_probability * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r3:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div>Overall Screening Level</div>
                    <div class="risk-number">
                        {risk_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -----------------------------------------------
        # REPORT TEXT
        # -----------------------------------------------

        if report_text:

            with st.expander(
                "📄 Extracted Report Text"
            ):

                st.write(report_text)

        if ocr_values:

            with st.expander(
                "🔍 Extracted Report Values"
            ):

                st.json(ocr_values)

        # -----------------------------------------------
        # AI SUGGESTIONS
        # -----------------------------------------------

        st.markdown(
            '<div class="section-title">🤖 Personalized Health Guidance</div>',
            unsafe_allow_html=True
        )

        patient_context = f"""
Patient age: {age}
Sex: {gender}

Diabetes screening risk:
{diabetes_probability * 100:.1f}%

Cardiovascular screening risk:
{cardio_probability * 100:.1f}%

Overall screening level:
{risk_level}

Symptoms:
{symptoms if symptoms else "No symptoms provided"}

Report information:
{report_text[:3000] if report_text else "No report text extracted."}
"""

        guidance_messages = [
            {
                "role": "system",
                "content": """
You are SehatAI.

Provide general patient education based on the
screening information.

Important:
- Do not diagnose.
- Do not prescribe medication.
- Do not provide medication dosage.
- Explain that ML risk scores are screening estimates.
- Recommend healthcare professional review.
- Use simple language.
- If emergency symptoms are mentioned, advise urgent
  medical evaluation.
"""
            },
            {
                "role": "user",
                "content": patient_context
            }
        ]

        with st.spinner(
            "Generating patient guidance..."
        ):

            guidance = ask_groq(
                guidance_messages
            )

        st.markdown(
            f"""
            <div class="info-card">
                {guidance.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------------------
        # SAVE TO SUPABASE
        # -----------------------------------------------

        if save_record is not None:

            try:

                save_record(
                    patient_name or "Anonymous",
                    age,
                    diabetes_probability * 100,
                    cardio_probability * 100,
                    risk_level
                )

                st.success(
                    "Screening record saved successfully."
                )

            except Exception:

                st.info(
                    "Screening completed. "
                    "History storage is not configured."
                )

        st.markdown(
            """
            <div class="disclaimer">
                <strong>Important:</strong>
                SehatAI is an AI-assisted screening and
                education prototype. It does not provide a
                medical diagnosis or replace a qualified
                healthcare professional.
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TAB 2 — REPORT ANALYSIS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">📄 Medical Report Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a PNG, JPG, JPEG or text-based PDF report."
    )

    report_file = st.file_uploader(
        "Upload report",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf"
        ],
        key="report_analysis"
    )

    if report_file is not None:

        if report_file.name.lower().endswith(".pdf"):

            text = extract_pdf_text(
                report_file
            )

            if text:

                st.success(
                    "PDF text extracted successfully."
                )

                with st.expander(
                    "View extracted text"
                ):

                    st.write(text)

            else:

                st.warning(
                    "No selectable text was found in this PDF. "
                    "If it is a scanned PDF, image OCR support "
                    "may be required."
                )

        else:

            try:

                image = Image.open(
                    io.BytesIO(
                        report_file.getvalue()
                    )
                )

                st.image(
                    image,
                    caption="Uploaded Report",
                    use_container_width=True
                )

            except Exception:

                st.error(
                    "Could not display this image."
                )

            if extract_report_values is not None:

                with st.spinner(
                    "Running OCR..."
                ):

                    values, text = process_report(
                        report_file
                    )

                if values:

                    st.success(
                        "Values extracted from report."
                    )

                    st.json(values)

                if text:

                    with st.expander(
                        "OCR Text"
                    ):

                        st.write(text)


# =========================================================
# TAB 3 — HOW IT WORKS
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">ℹ️ How SehatAI Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <h3>1️⃣ Patient Information</h3>

        The user enters basic patient information,
        symptoms and health measurements.

        </div>

        <div class="info-card">

        <h3>2️⃣ Machine Learning Screening</h3>

        SehatAI uses trained machine-learning models
        to estimate diabetes and cardiovascular risk.

        </div>

        <div class="info-card">

        <h3>3️⃣ Medical Report Processing</h3>

        Uploaded reports can be processed using
        PDF text extraction or OCR for images.

        </div>

        <div class="info-card">

        <h3>4️⃣ AI Patient Education</h3>

        Groq-hosted GPT-OSS provides general,
        understandable health guidance and can ask
        follow-up questions.

        </div>

        <div class="info-card">

        <h3>5️⃣ Human Review</h3>

        The system is designed as a decision-support
        and education prototype. Healthcare professionals
        should review important cases.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="disclaimer">

        <strong>Medical Safety Notice</strong><br><br>

        SehatAI is not a diagnostic system.
        AI-generated information may be incomplete or
        incorrect. It must not replace professional
        medical assessment.

        If someone has severe chest pain, severe breathing
        difficulty, fainting, confusion, seizure, severe
        bleeding, or another emergency symptom, seek
        emergency medical care immediately.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <br>
    <div style="text-align:center; color:#64748b; font-size:13px;">
        SehatAI • Rural Health Risk & Triage Copilot
        <br>
        AI-assisted screening • Patient education • Human review
    </div>
    """,
    unsafe_allow_html=True
)
