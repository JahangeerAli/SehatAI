import os
import re
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

try:
    import pytesseract
except Exception:
    pytesseract = None


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

    .stApp { background: #f5f8fc; }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .hero-box {
        background: linear-gradient(120deg, #087f8c, #0b9aaa, #2563eb);
        border-radius: 20px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
    }
    .hero-title { font-size: 38px; font-weight: 800; margin: 0; line-height: 1.1; }
    .hero-subtitle { font-size: 19px; font-weight: 600; margin-top: 7px; }
    .hero-description { font-size: 13px; margin-top: 8px; opacity: 0.95; }

    .section-title {
        font-size: 24px; font-weight: 750; color: #0f172a;
        margin-top: 12px; margin-bottom: 12px;
    }

    .simple-banner {
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 16px;
        color: #0e7490;
        font-size: 14px;
        line-height: 1.6;
    }

    .card {
        background: white; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(15,23,42,0.05);
    }
    .card-title { font-size: 17px; font-weight: 750; color: #0f172a; margin-bottom: 8px; }
    .card-text { color: #475569; font-size: 14px; line-height: 1.65; }

    .risk-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 20px; text-align: center; min-height: 145px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.05);
    }
    .risk-label { font-size: 14px; color: #64748b; font-weight: 600; }
    .risk-number { font-size: 32px; font-weight: 800; color: #0f766e; margin-top: 8px; }
    .risk-level { font-size: 24px; font-weight: 800; color: #2563eb; margin-top: 12px; }

    .notice {
        background: #eff6ff; border-left: 5px solid #2563eb; padding: 15px 18px;
        border-radius: 10px; color: #1e3a8a; margin: 15px 0; line-height: 1.6;
    }
    .warning {
        background: #fff7ed; border-left: 5px solid #f97316; padding: 16px 18px;
        border-radius: 10px; color: #7c2d12; margin: 15px 0; line-height: 1.6;
    }
    .emergency {
        background: #fef2f2; border-left: 5px solid #dc2626; padding: 16px 18px;
        border-radius: 10px; color: #7f1d1d; margin: 15px 0; line-height: 1.6;
    }

    section[data-testid="stSidebar"] { background: #f8fafc; }

    .chat-header {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white; padding: 15px; border-radius: 14px; margin-bottom: 12px;
    }
    .chat-header-title { font-size: 19px; font-weight: 750; }
    .chat-header-text { font-size: 12px; margin-top: 5px; opacity: 0.92; line-height: 1.5; }

    div[data-testid="stChatMessage"] { border-radius: 14px; }

    .stButton > button {
        border-radius: 10px;
    }

    .footer {
        text-align: center; color: #64748b; font-size: 12px; margin-top: 35px;
        padding-top: 20px; border-top: 1px solid #e2e8f0;
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
    <div class="hero-box">
        <div class="hero-title">🏥 SehatAI</div>
        <div class="hero-subtitle">Rural Health Risk &amp; Triage Copilot</div>
        <div class="hero-description">
            AI-assisted screening &nbsp;•&nbsp; Patient education &nbsp;•&nbsp; Human review
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
    language = st.selectbox("🌐 Language", ["English", "اردو", "Roman Urdu"])


def language_instruction():
    if language == "اردو":
        return "Respond in simple Urdu. Use easy words that ordinary patients can understand. Avoid complicated medical terminology."
    if language == "Roman Urdu":
        return "Respond in simple Roman Urdu. Use easy everyday language. Avoid complicated medical terminology."
    return "Respond in simple English. Use short, clear sentences. Avoid complicated medical terminology."


def t(en, ur, ro):
    """Small inline translation helper for UI labels."""
    if language == "اردو":
        return ur
    if language == "Roman Urdu":
        return ro
    return en


# =========================================================
# GROQ API KEY / CLIENT
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


def get_groq_client():
    key = get_groq_api_key()
    if not key or Groq is None:
        return None
    return Groq(api_key=key)


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
            max_completion_tokens=900,
        )

        answer = response.choices[0].message.content
        return answer.strip() if answer else "I could not generate a response right now."

    except Exception as e:
        error = str(e).lower()

        if "401" in error:
            return "The AI service key is invalid or expired. Please check GROQ_API_KEY in Streamlit Secrets."
        if "429" in error:
            return "The AI service is temporarily busy or rate-limited. Please try again after a short while."
        if "tool_use_failed" in error:
            return "The AI assistant encountered a temporary conversation error. Please send your message again."

        return "I cannot connect to the AI assistant right now. Please try again in a moment."


# =========================================================
# CHATBOT SYSTEM PROMPT
# =========================================================

def chatbot_system_prompt():
    return f"""
You are SehatAI, a warm and friendly patient education and triage assistant.

{language_instruction()}

Your role:
- Understand symptoms in plain, everyday language.
- Ask short, simple follow-up questions (2-3 at a time), one step at a time.
- Identify possible emergency warning signs.
- Give general health education, never overwhelming the patient with jargon.
- Suggest reasonable next steps.

You are NOT a doctor.

Never:
- Diagnose with certainty.
- Prescribe medicine or give medicine dosage.
- Tell the patient to stop prescribed medicine.
- Claim that an ML result proves a disease.

CONVERSATION STYLE:
Be conversational and empathetic, like a caring health worker, not a form.
When a patient gives only a symptom without enough information, do NOT give a
long lecture. Ask 2-3 short follow-up questions first, in very simple words.
Wait for the answer, then give organized guidance using these five short parts:
1. What it may mean generally
2. What they can do now
3. What to watch/monitor
4. When to contact a doctor
5. Emergency warning signs (only if relevant)

If the patient reports severe difficulty breathing, severe chest pain, fainting,
confusion, seizure, blue lips, severe bleeding, sudden one-sided weakness, or a
severe allergic reaction, clearly recommend urgent/emergency care immediately,
right at the top of your reply.

Keep every reply short, warm, and easy for someone with little formal education
to understand. Use simple analogies where helpful instead of medical terms.
"""


QUICK_SYMPTOMS = ["🤒 Fever", "🤧 Cough / Cold", "🤕 Headache", "🤢 Stomach pain", "😮‍💨 Breathing trouble"]


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        return ""

    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() for page in reader.pages]
        pages = [p for p in pages if p]
        return "\n".join(pages)
    except Exception:
        return ""


def extract_image_text(uploaded_file):
    """Fallback OCR using pytesseract if the custom OCR util is not available."""
    if pytesseract is None:
        return ""

    try:
        uploaded_file.seek(0)
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


# =========================================================
# SIMPLE LAB VALUE EXTRACTOR (regex fallback)
# Pulls common clinical numbers out of raw report text so the
# AI + ML models can use them, even without a dedicated OCR module.
# =========================================================

LAB_PATTERNS = {
    "Glucose": r"(?:glucose|blood\s*sugar|sugar\s*level)\D{0,12}(\d{2,3}(?:\.\d+)?)",
    "Cholesterol": r"(?:cholesterol|chol\.?)\D{0,12}(\d{2,3}(?:\.\d+)?)",
    "BMI": r"\bbmi\D{0,8}(\d{1,2}(?:\.\d+)?)",
    "Insulin": r"\binsulin\D{0,12}(\d{1,4}(?:\.\d+)?)",
    "MaximumHeartRate": r"(?:heart\s*rate|pulse)\D{0,12}(\d{2,3})",
    "Hemoglobin": r"(?:hemoglobin|haemoglobin|\bhb\b)\D{0,10}(\d{1,2}(?:\.\d+)?)",
}

BP_PATTERN = r"(?:blood\s*pressure|\bbp\b)\D{0,10}(\d{2,3})\s*/\s*(\d{2,3})"


def extract_lab_values_from_text(text):
    """Return a dict of best-guess clinical values found in free text."""
    if not text:
        return {}

    lowered = text.lower()
    values = {}

    for label, pattern in LAB_PATTERNS.items():
        match = re.search(pattern, lowered)
        if match:
            try:
                values[label] = float(match.group(1))
            except (ValueError, IndexError):
                pass

    bp_match = re.search(BP_PATTERN, lowered)
    if bp_match:
        try:
            values["Systolic"] = float(bp_match.group(1))
            values["Diastolic"] = float(bp_match.group(2))
        except (ValueError, IndexError):
            pass

    return values


# =========================================================
# DEFAULT (POPULATION-TYPICAL) VALUES
# Used to silently fill in advanced clinical fields the patient
# cannot reasonably know, in Simple Mode and in Report Analysis.
# =========================================================

DEFAULT_DIABETES = {
    "Pregnancies": 0,
    "Glucose": 100.0,
    "BloodPressure": 70.0,
    "SkinThickness": 20.0,
    "Insulin": 80.0,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 30,
}

DEFAULT_CARDIO = {
    "age": 30,
    "sex": 1,
    "cp": 0,
    "trestbps": 120.0,
    "chol": 200.0,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}


# =========================================================
# REPORT PROCESSING (image / pdf -> text + best-guess values)
# =========================================================

def process_report(uploaded_file):
    if uploaded_file is None:
        return {}, ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)
        values = extract_lab_values_from_text(text)
        return values, text

    # ---------------- IMAGE ----------------
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        text = ""
        values = {}

        if extract_report_values is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                image.save(temp.name)
                values, text = extract_report_values(temp.name)

        if not text:
            text = extract_image_text(uploaded_file)

        if not values:
            values = extract_lab_values_from_text(text)

        return values, text

    except Exception:
        return {}, ""


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    diabetes_model = joblib.load(os.path.join(APP_DIR, "models", "diabetes_model.pkl"))
    cardio_model = joblib.load(os.path.join(APP_DIR, "models", "cardio_model.pkl"))
    diabetes_features = joblib.load(os.path.join(APP_DIR, "models", "diabetes_features.pkl"))
    cardio_features = joblib.load(os.path.join(APP_DIR, "models", "cardio_features.pkl"))
    return diabetes_model, cardio_model, diabetes_features, cardio_features


try:
    diabetes_model, cardio_model, diabetes_features, cardio_features = load_models()
except Exception as e:
    st.error("⚠️ Model files could not be loaded.")
    st.code(str(e))
    st.stop()


def predict_health_risks(diabetes_data, cardio_data):
    """Align raw feature dicts to the trained model's expected columns and predict."""

    diabetes_input = {f: diabetes_data.get(f, 0) for f in diabetes_features}
    cardio_input = {f: cardio_data.get(f, 0) for f in cardio_features}

    diabetes_df = pd.DataFrame([diabetes_input])
    cardio_df = pd.DataFrame([cardio_input])

    try:
        if hasattr(diabetes_model, "predict_proba"):
            diabetes_risk = float(diabetes_model.predict_proba(diabetes_df)[0][1])
        else:
            diabetes_risk = float(diabetes_model.predict(diabetes_df)[0])
    except Exception as e:
        st.error("Diabetes model prediction failed.")
        st.code(str(e))
        diabetes_risk = 0.0

    try:
        if hasattr(cardio_model, "predict_proba"):
            cardio_risk = float(cardio_model.predict_proba(cardio_df)[0][1])
        else:
            cardio_risk = float(cardio_model.predict(cardio_df)[0])
    except Exception as e:
        st.error("Cardiovascular model prediction failed.")
        st.code(str(e))
        cardio_risk = 0.0

    return diabetes_risk, cardio_risk


def risk_level_label(overall):
    if overall >= 0.70:
        return "Higher"
    if overall >= 0.40:
        return "Moderate"
    return "Lower"


def render_risk_cards(diabetes_risk, cardio_risk, overall_level):
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f"""<div class="risk-card">
                <div class="risk-label">🩸 Diabetes Screening</div>
                <div class="risk-number">{diabetes_risk * 100:.1f}%</div>
                <div class="risk-label">Estimated screening risk</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f"""<div class="risk-card">
                <div class="risk-label">❤️ Cardiovascular Screening</div>
                <div class="risk-number">{cardio_risk * 100:.1f}%</div>
                <div class="risk-label">Estimated screening risk</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            f"""<div class="risk-card">
                <div class="risk-label">📌 Overall Screening Level</div>
                <div class="risk-level">{overall_level}</div>
                <div class="risk-label">Screening estimate</div>
            </div>""",
            unsafe_allow_html=True,
        )


# =========================================================
# SIDEBAR CHATBOT
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-header-title">🤖 SehatAI Health Assistant</div>
            <div class="chat-header-text">
                Tell me your symptoms. I will ask a couple of simple
                questions before giving you general guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chatbot_language = st.selectbox("Chat language", ["English", "اردو", "Roman Urdu"], key="chat_language")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋\n\n"
                    "I'm SehatAI. Tell me what you are experiencing, "
                    "or tap a quick option below.\n\n"
                    "For example: **I have fever.**"
                ),
            }
        ]

    if "pending_chat_input" not in st.session_state:
        st.session_state.pending_chat_input = None

    # ---- Quick reply chips for a more conversational feel ----
    st.caption(t("Quick start:", "فوری آپشن:", "Quick options:"))
    chip_cols = st.columns(len(QUICK_SYMPTOMS))
    for col, chip in zip(chip_cols, QUICK_SYMPTOMS):
        with col:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state.pending_chat_input = chip.split(" ", 1)[1]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_input = st.chat_input("Describe your symptoms...")

    user_message = st.session_state.pending_chat_input or typed_input
    st.session_state.pending_chat_input = None

    if user_message:
        st.session_state.chat_messages.append({"role": "user", "content": user_message})

        with st.chat_message("user"):
            st.markdown(user_message)

        if chatbot_language == "اردو":
            chat_language_instruction = "\nRespond in simple Urdu.\n"
        elif chatbot_language == "Roman Urdu":
            chat_language_instruction = "\nRespond in simple Roman Urdu.\n"
        else:
            chat_language_instruction = "\nRespond in simple English.\n"

        system_prompt = chatbot_system_prompt() + chat_language_instruction

        messages = [{"role": "system", "content": system_prompt}]
        for message in st.session_state.chat_messages[-12:]:
            messages.append({"role": message["role"], "content": message["content"]})

        with st.chat_message("assistant"):
            with st.spinner("SehatAI is thinking..."):
                answer = ask_groq(messages)
            st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. 👋\n\nTell me your symptoms whenever you are ready.",
            }
        ]
        st.rerun()


# =========================================================
# MAIN TABS
# =========================================================

tab_screening, tab_report, tab_about = st.tabs(
    ["🩺 Patient Screening", "📄 Report Analysis", "ℹ️ About SehatAI"]
)


# =========================================================
# PATIENT SCREENING
# =========================================================

with tab_screening:

    st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

    simple_mode = st.toggle(
        t("🧭 Simple Mode (easy questions for everyone)",
          "🧭 آسان موڈ (سب کے لیے آسان سوالات)",
          "🧭 Simple Mode (asaan sawal, sab k liye)"),
        value=True,
        help=t(
            "Turn this off only if you have exact lab test results to enter.",
            "اسے صرف اس وقت بند کریں جب آپ کے پاس درست لیب رپورٹ کے نمبر ہوں۔",
            "Ise sirf tab band karein jab aap k paas exact lab report k numbers hon.",
        ),
    )

    if simple_mode:
        st.markdown(
            f"""<div class="simple-banner">
            💡 {t(
                "Simple Mode asks only what most people already know about themselves. "
                "Anything technical (like lab-only values) is filled in automatically with a typical value.",
                "آسان موڈ صرف وہی سوالات پوچھتا ہے جو زیادہ تر لوگ خود جانتے ہیں۔ "
                "لیب ٹیسٹ والی تکنیکی ویلیوز خود بخود ایک عام اوسط نمبر سے پُر کر دی جاتی ہیں۔",
                "Simple Mode sirf wo sawal puchta hai jo zyada tar log khud jante hain. "
                "Lab test wali technical values automatically ek average number se bhar di jati hain."
            )}
            </div>""",
            unsafe_allow_html=True,
        )

    p1, p2, p3 = st.columns(3)

    with p1:
        patient_name = st.text_input(t("Patient Name", "مریض کا نام", "Patient ka naam"), placeholder=t("Enter patient name", "نام درج کریں", "Naam likhein"))
    with p2:
        age = st.number_input(t("Age", "عمر", "Umar"), min_value=1, max_value=120, value=30)
    with p3:
        sex_label = st.selectbox(t("Sex", "جنس", "Jins"), ["Male", "Female"])

    sex = 1 if sex_label == "Male" else 0
    extra_notes = []

    if simple_mode:

        # -----------------------------------------------
        # SIMPLE MODE — plain-language questions only
        # -----------------------------------------------

        st.markdown('<div class="section-title">🧍 Basic Details</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            weight_kg = st.number_input(t("Weight (kg)", "وزن (کلوگرام)", "Weight (kg)"), min_value=10.0, max_value=250.0, value=65.0)
        with s2:
            height_cm = st.number_input(t("Height (cm)", "قد (سینٹی میٹر)", "Height (cm)"), min_value=80.0, max_value=230.0, value=165.0)

        bmi = weight_kg / ((height_cm / 100) ** 2)

        pregnancies = 0
        if sex_label == "Female":
            pregnancies = st.number_input(
                t("How many times has she been pregnant? (0 if never / not applicable)",
                  "وہ کتنی بار حاملہ ہوئی ہیں؟ (کبھی نہیں تو 0 لکھیں)",
                  "Wo kitni baar pregnant hui hain? (kabhi nahi to 0)"),
                min_value=0, max_value=20, value=0,
            )

        st.markdown('<div class="section-title">🩸 What You Might Already Know</div>', unsafe_allow_html=True)

        g1, g2 = st.columns([1, 2])
        with g1:
            glucose_unknown = st.checkbox(t("I don't know my blood sugar", "مجھے شوگر لیول معلوم نہیں", "Mujhe sugar level pata nahi"), value=True)
        with g2:
            glucose = st.number_input(
                t("Blood sugar / glucose level (mg/dL)", "بلڈ شوگر / گلوکوز لیول", "Blood sugar / glucose level"),
                min_value=0.0, max_value=500.0, value=100.0, disabled=glucose_unknown,
            )
        if glucose_unknown:
            glucose = DEFAULT_DIABETES["Glucose"]

        bp1, bp2 = st.columns([1, 2])
        with bp1:
            bp_unknown = st.checkbox(t("I don't know my blood pressure", "مجھے بلڈ پریشر معلوم نہیں", "Mujhe blood pressure pata nahi"), value=True)
        with bp2:
            bp_text = st.text_input(
                t("Blood pressure (e.g. 120/80)", "بلڈ پریشر (مثلاً 120/80)", "Blood pressure (misal 120/80)"),
                placeholder="120/80", disabled=bp_unknown,
            )

        systolic, diastolic = DEFAULT_CARDIO["trestbps"], DEFAULT_DIABETES["BloodPressure"]
        if not bp_unknown and bp_text:
            match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", bp_text)
            if match:
                systolic, diastolic = float(match.group(1)), float(match.group(2))

        c1, c2 = st.columns([1, 2])
        with c1:
            chol_unknown = st.checkbox(t("I don't know my cholesterol", "مجھے کولیسٹرول معلوم نہیں", "Mujhe cholesterol pata nahi"), value=True)
        with c2:
            cholesterol = st.number_input(
                t("Cholesterol (mg/dL)", "کولیسٹرول", "Cholesterol"),
                min_value=0.0, max_value=700.0, value=200.0, disabled=chol_unknown,
            )
        if chol_unknown:
            cholesterol = DEFAULT_CARDIO["chol"]

        family_diabetes = st.radio(
            t("Does anyone in your close family have diabetes?", "کیا خاندان میں کسی کو شوگر کی بیماری ہے؟", "Khandan mein kisi ko sugar ki bimari hai?"),
            [t("No", "نہیں", "Nahi"), t("Yes", "ہاں", "Haan")], horizontal=True,
        )
        pedigree = 0.8 if family_diabetes in ("Yes", "ہاں", "Haan") else 0.2

        st.markdown('<div class="section-title">❤️ Heart Health</div>', unsafe_allow_html=True)

        chest_pain = st.radio(
            t("Do you get chest pain or discomfort?", "کیا آپ کو سینے میں درد ہوتا ہے؟", "Kya aap ko seene mein dard hota hai?"),
            [
                t("No pain", "کوئی درد نہیں", "Koi dard nahi"),
                t("Only during activity/exercise", "صرف کام یا ورزش کے دوران", "Sirf kaam ya exercise k dauran"),
                t("Even while resting", "آرام کے دوران بھی", "Aaram k dauran b"),
            ],
        )
        cp = {0: 0, 1: 1, 2: 2}[
            [
                t("No pain", "کوئی درد نہیں", "Koi dard nahi"),
                t("Only during activity/exercise", "صرف کام یا ورزش کے دوران", "Sirf kaam ya exercise k dauran"),
                t("Even while resting", "آرام کے دوران بھی", "Aaram k dauran b"),
            ].index(chest_pain)
        ]

        breathless = st.radio(
            t("Do you get very tired or breathless quickly during activity?", "کیا آپ کام کرتے ہوئے جلدی سانس پھول جاتی ہے؟", "Kaam k dauran jaldi saans phool jati hai?"),
            [t("No", "نہیں", "Nahi"), t("Yes", "ہاں", "Haan")], horizontal=True, key="breathless",
        )
        exang = 1 if breathless in ("Yes", "ہاں", "Haan") else 0

        fbs = 1 if glucose > 120 else 0

        st.markdown('<div class="section-title">🗒️ Anything Else?</div>', unsafe_allow_html=True)

        other_flags = st.multiselect(
            t("Select anything that applies (optional)", "جو بات لاگو ہو منتخب کریں (اختیاری)", "Jo baat lagu ho select karein (optional)"),
            [
                t("I smoke or use tobacco", "میں تمباکو نوشی کرتا/کرتی ہوں", "Main tambaku istemal karta/karti hun"),
                t("I have fainted or felt like fainting recently", "حال ہی میں بےہوش ہوا/ہوئی ہوں", "Recently behosh hua/hui hun"),
                t("I feel very thirsty and urinate often", "بہت پیاس لگتی ہے اور پیشاب زیادہ آتا ہے", "Bohat pyaas lagti hai aur peshab zyada aata hai"),
                t("I feel tired all the time", "ہمیشہ تھکاوٹ محسوس ہوتی ہے", "Hamesha thakawat mehsoos hoti hai"),
            ],
        )
        extra_notes.extend(other_flags)

        # Technical values not asked in Simple Mode — typical defaults
        skin_thickness = DEFAULT_DIABETES["SkinThickness"]
        insulin = DEFAULT_DIABETES["Insulin"]
        diabetes_age = age
        trestbps = systolic
        blood_pressure = diastolic
        chol = cholesterol
        restecg = DEFAULT_CARDIO["restecg"]
        thalach = DEFAULT_CARDIO["thalach"]
        oldpeak = DEFAULT_CARDIO["oldpeak"]
        slope = DEFAULT_CARDIO["slope"]
        ca = DEFAULT_CARDIO["ca"]
        thal = DEFAULT_CARDIO["thal"]

    else:

        # -----------------------------------------------
        # ADVANCED MODE — full clinical form (unchanged, with tooltips)
        # -----------------------------------------------

        st.markdown('<div class="section-title">🩸 Diabetes Screening</div>', unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        with d2:
            glucose = st.number_input("Glucose", min_value=0.0, max_value=500.0, value=100.0, help="Blood sugar level in mg/dL.")
        with d3:
            blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=300.0, value=70.0, help="Diastolic blood pressure (mm Hg).")
        with d4:
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=200.0, value=20.0, help="Triceps skin-fold thickness (mm) from a lab test.")

        d5, d6, d7, d8 = st.columns(4)
        with d5:
            insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, help="Serum insulin (mu U/ml) from a lab test.")
        with d6:
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        with d7:
            pedigree = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=5.0, value=0.47, help="A score for family history of diabetes.")
        with d8:
            diabetes_age = st.number_input("Diabetes Age", min_value=1, max_value=120, value=int(age))

        st.markdown('<div class="section-title">❤️ Cardiovascular Screening</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0=No pain, 1=Mild/activity pain, 2=Pain at rest, 3=Other")
        with c2:
            trestbps = st.number_input("Resting Blood Pressure", min_value=50.0, max_value=300.0, value=120.0, help="Systolic blood pressure (mm Hg) at rest.")
        with c3:
            chol = st.number_input("Cholesterol", min_value=50.0, max_value=700.0, value=200.0)
        with c4:
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
        with c6:
            thalach = st.number_input("Maximum Heart Rate", min_value=50.0, max_value=250.0, value=150.0)
        with c7:
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
        with c8:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)

        c9, c10, c11 = st.columns(3)
        with c9:
            slope = st.selectbox("ST Slope", [0, 1, 2])
        with c10:
            ca = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4])
        with c11:
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # =====================================================
    # SYMPTOMS (shown in both modes)
    # =====================================================

    st.markdown('<div class="section-title">📝 Symptoms</div>', unsafe_allow_html=True)

    symptoms = st.text_area(
        t("Describe symptoms in your own words", "اپنے الفاظ میں علامات بیان کریں", "Apne alfaaz mein symptoms batayein"),
        placeholder=t(
            "Example: I have fever since yesterday, temperature is 38.5°C, and I have a sore throat.",
            "مثال: مجھے کل سے بخار ہے، ٹمپریچر 38.5 ہے، اور گلے میں خراش ہے۔",
            "Misal: Mujhe kal se fever hai, temperature 38.5 hai, aur gale mein kharash hai.",
        ),
        height=100,
    )

    st.markdown('<div class="section-title">📄 Medical Report (optional)</div>', unsafe_allow_html=True)

    report = st.file_uploader(
        t("Upload PNG, JPG, JPEG or PDF", "PNG، JPG، JPEG یا PDF اپلوڈ کریں", "PNG, JPG, JPEG ya PDF upload karein"),
        type=["png", "jpg", "jpeg", "pdf"],
    )

    run_screening = st.button(
        t("🔎 Run Health Screening", "🔎 صحت کی جانچ کریں", "🔎 Health Screening Chalayen"),
        type="primary", use_container_width=True,
    )

    if run_screening:

        diabetes_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": diabetes_age,
        }

        cardio_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }

        ocr_values, report_text = process_report(report) if report else ({}, "")

        symptom_adjustments = {}
        if symptoms and parse_symptoms:
            try:
                symptom_adjustments = parse_symptoms(symptoms)
            except Exception:
                symptom_adjustments = {}

        for key, value in ocr_values.items():
            if key in diabetes_data:
                diabetes_data[key] = value
            if key in cardio_data:
                cardio_data[key] = value

        diabetes_risk, cardio_risk = predict_health_risks(diabetes_data, cardio_data)
        overall = (diabetes_risk + cardio_risk) / 2
        overall_level = risk_level_label(overall)

        st.markdown('<div class="section-title">📊 Your Screening Summary</div>', unsafe_allow_html=True)
        render_risk_cards(diabetes_risk, cardio_risk, overall_level)

        st.markdown('<div class="section-title">💡 What These Results Mean</div>', unsafe_allow_html=True)

        if language == "اردو":
            explanation = f"""
            <b>ذیابیطس اسکریننگ:</b> {diabetes_risk * 100:.1f}%<br>
            <b>دل اور خون کی نالیوں کی اسکریننگ:</b> {cardio_risk * 100:.1f}%<br>
            <b>مجموعی سطح:</b> {overall_level}<br><br>
            یہ نتائج صرف کمپیوٹر ماڈل کی اسکریننگ کا اندازہ ہیں۔ یہ بیماری کی تشخیص نہیں ہیں۔
            حتمی طبی فیصلہ ڈاکٹر یا qualified healthcare professional کرے۔
            """
        elif language == "Roman Urdu":
            explanation = f"""
            <b>Diabetes screening:</b> {diabetes_risk * 100:.1f}%<br>
            <b>Heart/cardiovascular screening:</b> {cardio_risk * 100:.1f}%<br>
            <b>Overall level:</b> {overall_level}<br><br>
            Ye results sirf computer model ki screening estimate hain. Ye kisi disease ki diagnosis nahi hain.
            Final medical decision qualified healthcare professional kare.
            """
        else:
            explanation = f"""
            <b>Diabetes screening:</b> {diabetes_risk * 100:.1f}%<br>
            <b>Cardiovascular screening:</b> {cardio_risk * 100:.1f}%<br>
            <b>Overall level:</b> {overall_level}<br><br>
            These results are estimates from a computer screening model. They are not a medical diagnosis.
            A qualified healthcare professional should review important cases.
            """

        st.markdown(f'<div class="card"><div class="card-text">{explanation}</div></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="notice">
                <strong>ℹ️ Important:</strong>
                A screening percentage is not the same thing as a confirmed probability of developing a disease.
                The result depends on the training data and model. Your healthcare professional should
                interpret it together with your medical history and examination.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if symptoms:
            st.markdown('<div class="section-title">📝 Symptoms You Reported</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-text">{html.escape(symptoms)}</div></div>', unsafe_allow_html=True)

        if extra_notes:
            st.markdown('<div class="section-title">🗒️ Additional Notes</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card"><div class="card-text">{html.escape(", ".join(extra_notes))}</div></div>',
                unsafe_allow_html=True,
            )

        if ocr_values:
            st.markdown('<div class="section-title">📄 Report Values Detected</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(list(ocr_values.items()), columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True,
            )

        if report_text:
            with st.expander(t("📄 View extracted report text", "📄 رپورٹ کا نکالا گیا متن دیکھیں", "📄 Report ka nikala gaya text dekhein")):
                st.write(report_text)

        st.markdown('<div class="section-title">🤖 SehatAI Guidance</div>', unsafe_allow_html=True)

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

Additional patient notes:
{", ".join(extra_notes) if extra_notes else "None."}

Report text:
{report_text[:2500] if report_text else "No report text available."}

Write using these headings:
### 1. Screening Summary
### 2. What This Means
### 3. Healthy Next Steps
### 4. About the Report
### 5. When to Contact a Doctor
### 6. Emergency Warning Signs

Do not diagnose. Do not prescribe medicine. Do not give medicine dosage.
Do not say the patient definitely has or does not have a disease.

{language_instruction()}
"""

        with st.spinner(t("🤖 Preparing your health guidance...", "🤖 آپ کی رپورٹ تیار کی جا رہی ہے...", "🤖 Aap ki guidance tayyar ki ja rahi hai...")):
            guidance = ask_groq(
                [
                    {"role": "system", "content": "You are a careful health education assistant. Your output must be organized, concise and patient-friendly. Never diagnose or prescribe."},
                    {"role": "user", "content": guidance_prompt},
                ]
            )

        st.markdown(f'<div class="card"><div class="card-text">{guidance.replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="emergency">
                <strong>🚨 Emergency warning</strong><br><br>
                If you or the patient has severe difficulty breathing, severe chest pain, fainting, confusion,
                seizure, blue lips, severe bleeding, sudden weakness on one side, or another serious emergency
                symptom, seek emergency medical care immediately.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if save_record is not None:
            try:
                save_record(patient_name or "Anonymous", age, diabetes_risk * 100, cardio_risk * 100, overall_level)
                st.success("✅ Screening record saved.")
            except Exception:
                st.info("Screening completed. History storage is not currently configured.")


# =========================================================
# REPORT ANALYSIS TAB
# =========================================================

with tab_report:

    st.markdown('<div class="section-title">📄 Medical Report Analyzer</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Upload a medical report</div>
            <div class="card-text">
                SehatAI reads the text from your PDF or image report, pulls out any
                recognizable numbers (like glucose or cholesterol), runs them through
                the trained screening models, and explains everything in plain language.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    standalone_report = st.file_uploader(
        t("Choose a report", "رپورٹ منتخب کریں", "Report select karein"),
        type=["png", "jpg", "jpeg", "pdf"], key="standalone_report",
    )

    if standalone_report:

        filename = standalone_report.name.lower()

        if not filename.endswith(".pdf"):
            try:
                image = Image.open(io.BytesIO(standalone_report.getvalue()))
                st.image(image, caption="Uploaded Medical Report", use_container_width=True)
            except Exception:
                st.error("Unable to display this image.")

        with st.spinner(t("🔍 Reading report...", "🔍 رپورٹ پڑھی جا رہی ہے...", "🔍 Report parhi ja rahi hai...")):
            values, text = process_report(standalone_report)

        if text:
            st.success(t("✅ Text extracted successfully.", "✅ متن کامیابی سے نکال لیا گیا۔", "✅ Text kamyabi se nikal liya gaya."))
            with st.expander(t("View extracted text", "متن دیکھیں", "Text dekhein")):
                st.write(text)
        else:
            st.warning(
                t(
                    "No text could be read from this file. If it is a scanned image, "
                    "make sure OCR is enabled (see the setup guide for adding pytesseract).",
                    "اس فائل سے متن نہیں پڑھا جا سکا۔ اگر یہ اسکین شدہ تصویر ہے تو OCR فعال کریں۔",
                    "Is file se text nahi parha ja saka. Agar ye scanned image hai to OCR enable karein.",
                )
            )

        if values:
            st.markdown('<div class="section-title">🔢 Values Detected in Report</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(list(values.items()), columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True,
            )

        # ---- Run the trained ML models on whatever values we found ----
        diabetes_data = dict(DEFAULT_DIABETES)
        cardio_data = dict(DEFAULT_CARDIO)

        if "Glucose" in values:
            diabetes_data["Glucose"] = values["Glucose"]
            cardio_data["fbs"] = 1 if values["Glucose"] > 120 else 0
        if "BMI" in values:
            diabetes_data["BMI"] = values["BMI"]
        if "Insulin" in values:
            diabetes_data["Insulin"] = values["Insulin"]
        if "Cholesterol" in values:
            cardio_data["chol"] = values["Cholesterol"]
        if "MaximumHeartRate" in values:
            cardio_data["thalach"] = values["MaximumHeartRate"]
        if "Systolic" in values:
            cardio_data["trestbps"] = values["Systolic"]
        if "Diastolic" in values:
            diabetes_data["BloodPressure"] = values["Diastolic"]

        has_enough_signal = any(
            k in values for k in ("Glucose", "Cholesterol", "BMI", "Systolic", "Insulin", "MaximumHeartRate")
        )

        if has_enough_signal:
            diabetes_risk, cardio_risk = predict_health_risks(diabetes_data, cardio_data)
            overall_level = risk_level_label((diabetes_risk + cardio_risk) / 2)

            st.markdown('<div class="section-title">📊 Model-Based Screening (from report)</div>', unsafe_allow_html=True)
            render_risk_cards(diabetes_risk, cardio_risk, overall_level)
            st.caption(
                t(
                    "Any value not found in the report was filled in with a typical average, so treat this as a rough estimate.",
                    "رپورٹ میں نہ ملنے والی ویلیوز ایک عام اوسط سے پُر کی گئی ہیں، اس لیے یہ صرف ایک اندازہ ہے۔",
                    "Report mein na milne wali values ek average number se bhari gayi hain, is liye ye sirf ek andaza hai.",
                )
            )

        if text:
            st.markdown('<div class="section-title">🤖 AI Analysis of This Report</div>', unsafe_allow_html=True)

            analysis_prompt = f"""
A patient uploaded a medical report. Here is the text extracted from it
(it may be incomplete or contain OCR errors):

---
{text[:3000]}
---

Detected numeric values (may be empty): {values if values else "None detected"}

Please:
1. Summarize in plain language what this report seems to show.
2. Point out any values that look unusually high or low, in simple terms.
3. Give general, non-diagnostic health guidance based on this.
4. Clearly state this is not a diagnosis and a doctor should review the full report.

{language_instruction()}
"""

            with st.spinner(t("🤖 Analyzing report...", "🤖 رپورٹ کا تجزیہ ہو رہا ہے...", "🤖 Report ka analysis ho raha hai...")):
                analysis = ask_groq(
                    [
                        {"role": "system", "content": "You are a careful medical report explainer for patients. Never diagnose or prescribe. Be clear and reassuring but honest."},
                        {"role": "user", "content": analysis_prompt},
                    ]
                )

            st.markdown(f'<div class="card"><div class="card-text">{analysis.replace(chr(10), "<br>")}</div></div>', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="warning">
                    <strong>⚠️ Note:</strong> This analysis is AI-generated from extracted text and may
                    miss details or misread numbers. Always share the original report with a doctor.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# ABOUT TAB
# =========================================================

with tab_about:

    st.markdown('<div class="section-title">ℹ️ About SehatAI</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🏥 What is SehatAI?</div>
            <div class="card-text">
                SehatAI is an AI-assisted health screening and patient education prototype
                designed for resource-constrained and rural healthcare workflows.
            </div>
        </div>

        <div class="card">
            <div class="card-title">🧠 Machine Learning</div>
            <div class="card-text">
                The prototype uses trained machine-learning models for diabetes and
                cardiovascular screening, plus a free Groq-hosted language model for
                conversation and report explanations.
            </div>
        </div>

        <div class="card">
            <div class="card-title">📄 Report Processing</div>
            <div class="card-text">
                Text-based PDF reports and image reports can be processed to extract
                useful values, which are then run through the same screening models
                and explained in plain language.
            </div>
        </div>

        <div class="card">
            <div class="card-title">🤖 AI Assistant</div>
            <div class="card-text">
                The SehatAI assistant can have a conversational symptom discussion,
                ask follow-up questions, and provide general health education.
            </div>
        </div>

        <div class="warning">
            <strong>⚠️ Medical Safety Notice</strong><br><br>
            SehatAI is a research and decision-support prototype. It is not a doctor and
            does not provide a medical diagnosis or prescription. AI-generated information
            can be incomplete or incorrect. Important decisions should be made with a
            qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🏥 <strong>SehatAI</strong> — Rural Health Risk &amp; Triage Copilot
        <br>
        AI-assisted screening • Patient education • Human review
    </div>
    """,
    unsafe_allow_html=True,
)
