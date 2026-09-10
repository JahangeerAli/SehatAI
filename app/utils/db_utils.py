"""Supabase database utilities — save and fetch patient screening records."""

import streamlit as st
from supabase import create_client
from datetime import datetime

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)


def save_record(name, age, diabetes_risk, cardio_risk, level):
    supabase.table("patient_records").insert({
        "name": name,
        "age": age,
        "diabetes_risk": float(diabetes_risk),
        "cardio_risk": float(cardio_risk),
        "risk_level": level,
        "created_at": datetime.utcnow().isoformat()
    }).execute()


def fetch_records():
    response = (
        supabase.table("patient_records")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
