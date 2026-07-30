import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from fpdf import FPDF
import datetime
import smtplib
from email.message import EmailMessage
import os
import google.generativeai as genai

# 🌟 Gemini API setup
genai.configure(api_key=st.secrets["gemini"]["api_key"])

# 📩 Email sender
def send_pdf_via_email(receiver_email, pdf_path):
    sender = st.secrets["email"]["sender_email"]
    password = st.secrets["email"]["sender_password"]

    msg = EmailMessage()
    msg["Subject"] = "Your Medical Report"
    msg["From"] = sender
    msg["To"] = receiver_email
    msg.set_content("Please find your attached medical report.")

    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)

# 🧠 Load ML models
load_diabetes = pickle.load(open("diabetes.pkl", 'rb'))
load_heart = pickle.load(open("heart.pkl", 'rb'))
load_breast = pickle.load(open("breast.pkl", 'rb'))

st.set_page_config(page_title="AI Medical Predictor", page_icon="🩺", layout="wide")

# ---- Custom CSS for dark theme ----
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .report-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4fc3f7;
        }
        .sub-heading {
            font-size: 1.2rem;
            color: #81d4fa;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Banner
# st.image("https://img.freepik.com/premium-vector/doctor-medical-technology-health-care-icons-virtual-screen-hospital-medicine-concept_670147-177.jpg", use_column_width=True)

# User Info Section
if "user_details" not in st.session_state:
    st.session_state["user_details"] = {}
    st.session_state["details_filled"] = False

if not st.session_state["details_filled"]:
    with st.container():
        st.markdown("<div class='report-title'>🧾 Enter Your Basic Information</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Full Name")
            age = st.number_input("🎂 Age", min_value=1, max_value=120)
            gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
            date = st.date_input("📅 Date", datetime.date.today())
        with col2:
            city = st.text_input("🏙️ City")
            state = st.text_input("🌍 State")
            email = st.text_input("✉️ Email Address")

        if st.button("✅ Save & Proceed"):
            if name and city and state and email:
                st.session_state["user_details"] = {
                    "name": name,
                    "age": int(age),
                    "gender": gender,
                    "date": date.strftime('%Y-%m-%d'),
                    "city": city,
                    "state": state,
                    "email": email
                }
                st.session_state["details_filled"] = True
                st.rerun()
            else:
                st.warning("⚠️ Please fill all required fields before proceeding.")
    st.stop()

# Sidebar Chatbot
with st.sidebar:
    st.markdown("<div style='font-size:1.5rem;font-weight:bold;color:#007B8A;'>🧠 AI Health Assistant</div>", unsafe_allow_html=True)
    st.write("Ask anything related to health...")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area("Your Question", key="gemini_user_input")
    if st.button("Ask Our Chatbot"):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking..."):
                try:
                    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                    chat = model.start_chat(history=[])
                    response = chat.send_message(user_input)
                    reply = response.text
                    st.session_state.chat_history.append(("ai", reply))
                    st.success("AI Response:")
                    st.write(reply)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    st.markdown("---")
    for role, msg in reversed(st.session_state.chat_history):
        st.markdown(f"**{'You' if role == 'user' else 'AI'}:** {msg}")

# Menu
selected = option_menu(
    menu_title="Disease Prediction",
    options=["Diabetes Prediction", "Heart Disease Prediction", "Breast Cancer Prediction"],
    icons=["activity", "heart", "cast"],
    default_index=0,
    orientation="horizontal"
)

# PDF Generator
def generate_report(disease, diagnosis, parameters):
    user = st.session_state["user_details"]
    prompt = f"""
You are a medical assistant generating a personalized report based on a patient's test inputs.

Patient Info:
- Name: {user.get('name')}
- Age: {user.get('age')}
- Gender: {user.get('gender')}
- Disease: {disease}
- Diagnosis Result: {diagnosis}

Patient’s Health Parameters:
"""
    for param, value in parameters.items():
        prompt += f"\n- {param}: {value}"

    prompt += """

Please generate a short and simple medical report in the following structure:

1. 📝 Report Summary:
- Explain what the diagnosis means in simple, non-technical English.
- Assume the reader has little to no medical knowledge.

2. 🥗 Diet Suggestions:
- Mention a basic daily diet plan with timings (morning, afternoon, evening, etc.).
- Explain what each food helps with in very simple words.

3. 💡 AI Suggestions:
- Give 3 tips or habits to follow to improve health based on the disease.
- Keep it short and clearly written.

4. 🔍 Conclusion & Follow-up:
- Mention 2 next steps or checkups the patient should do.
- Keep tone positive and motivating.

Rules:
- Use very easy and short English sentences.
- Do not include any extra technical explanation.
- This report should be friendly and readable by non-medical people.
"""

    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        chat = model.start_chat(history=[])
        suggestions = chat.send_message(prompt).text
    except Exception as e:
        suggestions = f"Could not fetch AI suggestions. Error: {e}"

    import re
    def clean_text(text): return text.encode('latin-1', 'ignore').decode('latin-1')
    suggestions = re.sub(r'\*+', '', suggestions)
    safe_suggestions = clean_text(suggestions)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Medical Test Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    for k, v in user.items():
        pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{disease} Test Results", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in parameters.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Final Diagnosis: {diagnosis}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, safe_suggestions)
    filename = f"{disease}_Report_{user['name'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename


# 🧠 Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.markdown("<div class='report-title'>🩸 Diabetes Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-heading'>Enter the following medical parameters:</div>", unsafe_allow_html=True)

    default_values = {
        "Pregnancies": 0, "Glucose": 0, "Blood Pressure": 0, "Skin Thickness": 0,
        "Insulin": 0, "BMI": 0.0, "Diabetes Pedigree Function": 0.0, "Age": st.session_state["user_details"]["age"]
    }
    sample_values = {
        "Pregnancies": 6, "Glucose": 148, "Blood Pressure": 72, "Skin Thickness": 35,
        "Insulin": 0, "BMI": 33.6, "Diabetes Pedigree Function": 0.627, "Age": st.session_state["user_details"]["age"]
    }

    if "diabetes_inputs" not in st.session_state:
        st.session_state.diabetes_inputs = default_values.copy()

    if st.button("🔄 Auto-Fill Sample Values"):
        st.session_state.diabetes_inputs = sample_values.copy()

    col1, col2 = st.columns(2)
    with col1:
        for key in list(default_values.keys())[:4]:
            st.session_state.diabetes_inputs[key] = st.number_input(
                key, value=st.session_state.diabetes_inputs[key], key=f"dia_{key}")
    with col2:
        for key in list(default_values.keys())[4:]:
            st.session_state.diabetes_inputs[key] = st.number_input(
                key, value=st.session_state.diabetes_inputs[key], key=f"dia_{key}")

    if st.button("🧬 Predict Diabetes", use_container_width=True):
        input_values = list(st.session_state.diabetes_inputs.values())
        result = load_diabetes.predict([input_values])[0]
        diagnosis = 'Diabetic' if result == 1 else 'Non-Diabetic'

        st.markdown("""
            <div style='background-color:#e8f9f7;padding:1.2rem;border-radius:10px;margin-top:1rem;'>
                <h3 style='color:#007B8A;'>🧾 Prediction Result: <span style='color:#003b4a;'>%s</span></h3>
            </div>
        """ % diagnosis, unsafe_allow_html=True)

        with st.expander("📄 Generate PDF Report & Email"):
            from fpdf import FPDF
            import re

            def clean_text(text):
                return text.encode('latin-1', 'ignore').decode('latin-1')

            user = st.session_state["user_details"]
            parameters = st.session_state.diabetes_inputs
            prompt = f"""
                You are a medical assistant generating a personalized report based on a patient's test inputs.
                Name: {user['name']}
                Age: {user['age']}
                Gender: {user['gender']}
                Disease: Diabetes
                Diagnosis: {diagnosis}
                Parameters: {parameters}
                Provide summary, diet plan, 3 tips, and 2 follow-up suggestions.
                Use simple English.
            """
            try:
                model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                chat = model.start_chat(history=[])
                suggestions = chat.send_message(prompt).text
            except Exception as e:
                suggestions = f"Could not fetch AI suggestions. Error: {e}"

            suggestions = re.sub(r'\*+', '', suggestions)
            suggestions = clean_text(suggestions)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Medical Test Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(5)
            for k, v in user.items():
                pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Diabetes Test Results", ln=True)
            pdf.set_font("Arial", size=12)
            for k, v in parameters.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Final Diagnosis: {diagnosis}", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, suggestions)

            filename = f"Diabetes_Report_{user['name'].replace(' ', '_')}.pdf"
            pdf.output(filename)

            with open(filename, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=filename)

            try:
                sender = st.secrets["email"]["sender_email"]
                password = st.secrets["email"]["sender_password"]

                msg = EmailMessage()
                msg["Subject"] = "Your Medical Report"
                msg["From"] = sender
                msg["To"] = user["email"]
                msg.set_content("Please find your attached medical report.")
                with open(filename, "rb") as f:
                    msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(msg)
                st.success("📧 Report sent to your email successfully!")
            except Exception as e:
                st.error(f"❌ Email failed to send: {e}")


# ❤️ Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.markdown("<div class='report-title'>❤️ Heart Disease Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-heading'>Provide the following heart health metrics:</div>", unsafe_allow_html=True)

    default_heart = {
        "Age": st.session_state["user_details"]["age"], "Sex": 0, "Chest Pain Type": 0,
        "Resting BP": 0, "Cholesterol": 0, "FBS > 120": 0, "Rest ECG": 0,
        "Max Heart Rate": 0, "Exercise Angina": 0, "Oldpeak": 0.0, "Slope": 0,
        "CA": 0, "Thal": 0
    }
    sample_heart = {
        "Age": st.session_state["user_details"]["age"], "Sex": 1, "Chest Pain Type": 3,
        "Resting BP": 130, "Cholesterol": 250, "FBS > 120": 0, "Rest ECG": 1,
        "Max Heart Rate": 150, "Exercise Angina": 0, "Oldpeak": 1.0, "Slope": 2,
        "CA": 0, "Thal": 2
    }

    if "heart_inputs" not in st.session_state:
        st.session_state.heart_inputs = default_heart.copy()

    if st.button("🔄 Auto-Fill Sample Values"):
        st.session_state.heart_inputs = sample_heart.copy()

    col1, col2 = st.columns(2)
    with col1:
        for k in list(default_heart.keys())[:7]:
            st.session_state.heart_inputs[k] = st.number_input(k, value=st.session_state.heart_inputs[k], key=f"heart_{k}")
    with col2:
        for k in list(default_heart.keys())[7:]:
            st.session_state.heart_inputs[k] = st.number_input(k, value=st.session_state.heart_inputs[k], key=f"heart_{k}")

    if st.button("🧬 Predict Heart Disease", use_container_width=True):
        input_vals = list(st.session_state.heart_inputs.values())
        result = load_heart.predict([input_vals])[0]
        diagnosis = 'Heart Disease Detected' if result == 1 else 'No Heart Disease'

        st.markdown("""
            <div style='background-color:#e8f9f7;padding:1.2rem;border-radius:10px;margin-top:1rem;'>
                <h3 style='color:#007B8A;'>🧾 Prediction Result: <span style='color:#003b4a;'>%s</span></h3>
            </div>
        """ % diagnosis, unsafe_allow_html=True)

        with st.expander("📄 Generate PDF Report & Email"):
            from fpdf import FPDF
            import re

            def clean_text(text):
                return text.encode('latin-1', 'ignore').decode('latin-1')

            user = st.session_state["user_details"]
            parameters = st.session_state.heart_inputs
            prompt = f"""
                You are a medical assistant creating a report for a heart patient.
                Name: {user['name']}
                Age: {user['age']}
                Gender: {user['gender']}
                Disease: Heart Disease
                Diagnosis: {diagnosis}
                Parameters: {parameters}
                Provide summary, diet plan, 3 tips, and 2 follow-up suggestions.
                Use simple English.
            """
            try:
                model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                chat = model.start_chat(history=[])
                suggestions = chat.send_message(prompt).text
            except Exception as e:
                suggestions = f"Could not fetch AI suggestions. Error: {e}"

            suggestions = re.sub(r'\*+', '', suggestions)
            suggestions = clean_text(suggestions)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Medical Test Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(5)
            for k, v in user.items():
                pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Heart Disease Test Results", ln=True)
            pdf.set_font("Arial", size=12)
            for k, v in parameters.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Final Diagnosis: {diagnosis}", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, suggestions)

            filename = f"Heart_Report_{user['name'].replace(' ', '_')}.pdf"
            pdf.output(filename)

            with open(filename, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=filename)

            try:
                sender = st.secrets["email"]["sender_email"]
                password = st.secrets["email"]["sender_password"]

                msg = EmailMessage()
                msg["Subject"] = "Your Medical Report"
                msg["From"] = sender
                msg["To"] = user["email"]
                msg.set_content("Please find your attached medical report.")
                with open(filename, "rb") as f:
                    msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(msg)
                st.success("📧 Report sent to your email successfully!")
            except Exception as e:
                st.error(f"❌ Email failed to send: {e}")

# 🎗️ Breast Cancer Prediction Page
if selected == "Breast Cancer Prediction":
    st.markdown("<div class='report-title'>🎗️ Breast Cancer Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-heading'>Please enter the following features:</div>", unsafe_allow_html=True)

    features = [
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error", "smoothness error",
        "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
        "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
    ]
    autofill_values = [14.0, 20.0, 90.0, 600.0, 0.1, 0.2, 0.2, 0.1, 0.2, 0.06,
                       1.0, 0.9, 6.0, 100.0, 0.007, 0.02, 0.03, 0.01, 0.02, 0.003,
                       15.0, 25.0, 100.0, 800.0, 0.14, 0.3, 0.4, 0.2, 0.3, 0.1]

    if "bc_inputs" not in st.session_state:
        st.session_state.bc_inputs = {feat: 0.0 for feat in features}

    if st.button("🔄 Auto-Fill Sample Values"):
        for i, feat in enumerate(features):
            st.session_state.bc_inputs[feat] = autofill_values[i]

    col1, col2 = st.columns(2)
    with col1:
        for feat in features[:15]:
            st.session_state.bc_inputs[feat] = st.number_input(feat, value=st.session_state.bc_inputs[feat], key=feat)
    with col2:
        for feat in features[15:]:
            st.session_state.bc_inputs[feat] = st.number_input(feat, value=st.session_state.bc_inputs[feat], key=feat)

    if st.button("🧬 Predict Breast Cancer", use_container_width=True):
        input_vals = list(st.session_state.bc_inputs.values())
        result = load_breast.predict([input_vals])[0]
        diagnosis = 'Malignant (Cancerous)' if result == 1 else 'Benign (Non-Cancerous)'

        st.markdown("""
            <div style='background-color:#e8f9f7;padding:1.2rem;border-radius:10px;margin-top:1rem;'>
                <h3 style='color:#007B8A;'>🧾 Prediction Result: <span style='color:#003b4a;'>%s</span></h3>
            </div>
        """ % diagnosis, unsafe_allow_html=True)

        with st.expander("📄 Generate PDF Report & Email"):
            from fpdf import FPDF
            import re

            def clean_text(text):
                return text.encode('latin-1', 'ignore').decode('latin-1')

            user = st.session_state["user_details"]
            parameters = st.session_state.bc_inputs
            prompt = f"""
                You are a medical assistant creating a report for a breast cancer patient.
                Name: {user['name']}
                Age: {user['age']}
                Gender: {user['gender']}
                Disease: Breast Cancer
                Diagnosis: {diagnosis}
                Parameters: {parameters}
                Provide summary, diet plan, 3 tips, and 2 follow-up suggestions.
                Use simple English.
            """
            try:
                model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
                chat = model.start_chat(history=[])
                suggestions = chat.send_message(prompt).text
            except Exception as e:
                suggestions = f"Could not fetch AI suggestions. Error: {e}"

            suggestions = re.sub(r'\*+', '', suggestions)
            suggestions = clean_text(suggestions)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Medical Test Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(5)
            for k, v in user.items():
                pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Breast Cancer Test Results", ln=True)
            pdf.set_font("Arial", size=12)
            for k, v in parameters.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Final Diagnosis: {diagnosis}", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, suggestions)

            filename = f"BreastCancer_Report_{user['name'].replace(' ', '_')}.pdf"
            pdf.output(filename)

            with open(filename, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=filename)

            try:
                sender = st.secrets["email"]["sender_email"]
                password = st.secrets["email"]["sender_password"]

                msg = EmailMessage()
                msg["Subject"] = "Your Medical Report"
                msg["From"] = sender
                msg["To"] = user["email"]
                msg.set_content("Please find your attached medical report.")
                with open(filename, "rb") as f:
                    msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender, password)
                    smtp.send_message(msg)
                st.success("📧 Report sent to your email successfully!")
            except Exception as e:
                st.error(f"❌ Email failed to send: {e}")
