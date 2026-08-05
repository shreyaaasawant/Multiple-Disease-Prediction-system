"""
MediPredict — core business logic.

Pure, testable functions for the MediPredict screening platform.
This module does NOT import Streamlit, so it can be unit-tested directly.
"""
import os
import pickle
import re
import warnings

import google.generativeai as genai
from fpdf import FPDF

# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "diabetes": "diabetes.pkl",
    "heart": "heart.pkl",
    "breast": "breast.pkl",
}

GEMINI_MODEL = "gemini-3.1-flash-lite"

_configured = False


def configure_gemini(api_key):
    """Configure the Gemini client (idempotent)."""
    global _configured
    if api_key and not _configured:
        genai.configure(api_key=api_key)
        _configured = True


def load_models(base_dir="."):
    """
    Load the three scikit-learn models from the given directory.
    Returns a dict of name -> model.
    """
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file missing: {fname}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        loaded[name] = model
    return loaded


# ---------------------------------------------------------------------------
#  Text helpers
# ---------------------------------------------------------------------------
def clean_text(text):
    """Strip characters that are not representable in latin-1 (FPDF requirement)."""
    return text.encode("latin-1", "ignore").decode("latin-1")


def risk_meta(probability):
    """Map a predicted probability (0..1) to a risk level + CSS class + label."""
    if probability < 0.35:
        return "low", "risk-low", "Low Risk"
    if probability < 0.65:
        return "moderate", "risk-moderate", "Moderate Risk"
    return "high", "risk-high", "High Risk"


# ---------------------------------------------------------------------------
#  Gemini AI
# ---------------------------------------------------------------------------
def get_gemini_suggestions(prompt):
    """Wrap Gemini generation with graceful error handling."""
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        chat = model.start_chat(history=[])
        return chat.send_message(prompt).text
    except Exception as exc:
        return f"Could not fetch AI suggestions. Error: {exc}"


# ---------------------------------------------------------------------------
#  PDF report generation
# ---------------------------------------------------------------------------
def build_pdf(user, disease, diagnosis, parameters, suggestions):
    """Generate a professional PDF report and return its filename."""
    suggestions = re.sub(r"\*+", "", suggestions)
    suggestions = clean_text(suggestions)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.ln(6)
    pdf.cell(0, 10, "MediPredict - Medical Test Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.ln(8)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in user.items():
        pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"{disease} Test Results", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in parameters.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Final Diagnosis: {diagnosis}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, suggestions)
    filename = f"{disease}_Report_{user['name'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename


# ---------------------------------------------------------------------------
#  Predictions
# ---------------------------------------------------------------------------
def predict_diabetes(model, values):
    """Run the diabetes model and return (result, probability)."""
    proba = model.predict_proba([values])[0]
    result = int(model.predict([values])[0])
    probability = float(proba[1]) if result == 1 else float(proba[0])
    return result, probability


def predict_heart(model, values):
    proba = model.predict_proba([values])[0]
    result = int(model.predict([values])[0])
    probability = float(proba[1]) if result == 1 else float(proba[0])
    return result, probability


def predict_breast(model, values):
    proba = model.predict_proba([values])[0]
    result = int(model.predict([values])[0])
    probability = float(proba[1]) if result == 1 else float(proba[0])
    return result, probability

