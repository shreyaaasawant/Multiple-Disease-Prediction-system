"""
MediPredict — ML-Powered Disease Prediction Platform
=====================================================
A premium, production-quality healthcare screening dashboard.

 - Disease predictions are performed by scikit-learn Logistic Regression models.
 - Personalized explanations & conversational assistant are powered by Google Gemini.
 - Professional glassmorphism UI with Lucide-style SVG icons.

For educational & research purposes only. Not a substitute for medical advice.
"""

import datetime
import smtplib
import warnings
from email.message import EmailMessage

import streamlit as st
from streamlit_option_menu import option_menu

# Core business logic (models, predictions, PDF, Gemini) lives in core.py,
# which is fully unit-testable and free of Streamlit side effects.
import core

# ---------------------------------------------------------------------------
#  Page configuration  (MUST be the very first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediPredict — ML Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Suppress the InconsistentVersionWarning from models trained on older sklearn
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
#  Google Gemini configuration
# ---------------------------------------------------------------------------
try:
    core.configure_gemini(st.secrets["gemini"]["api_key"])
except Exception:
    # App must still boot if the key is missing; chat/explanation will show a friendly note.
    pass

# ---------------------------------------------------------------------------
#  Load ML models (cached for the session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction models…")
def load_models():
    """Load the three scikit-learn models once and cache them."""
    try:
        return core.load_models()
    except FileNotFoundError:
        st.error("Model files (.pkl) not found. Please ensure they are in the project root.")
        return {}


MODELS = load_models()

# ===========================================================================
#  GLOBAL DESIGN SYSTEM  (handcrafted, glassmorphism, premium)
# ===========================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg:              #f6f8fb;
    --bg-grad-1:       #eaf1fb;
    --bg-grad-2:       #f6f8fb;
    --surface:         rgba(255,255,255,0.72);
    --surface-solid:   #ffffff;
    --border:          rgba(15,23,42,0.08);
    --border-strong:   #e6e9f0;
    --text:            #0f172a;
    --text-2:          #64748b;
    --text-3:          #94a3b8;
    --primary:         #2563eb;
    --primary-2:       #0ea5e9;
    --primary-soft:    #eff6ff;
    --success:         #16a34a;
    --success-soft:    #f0fdf4;
    --warning:         #d97706;
    --warning-soft:    #fffbeb;
    --danger:          #dc2626;
    --danger-soft:     #fef2f2;
    --radius:          16px;
    --radius-sm:       12px;
    --shadow-sm:       0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06);
    --shadow-md:       0 4px 16px rgba(15,23,42,0.08);
    --shadow-lg:       0 12px 40px rgba(15,23,42,0.12);
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font);
    color: var(--text);
}

/* ---- App background ---- */
.stApp {
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(37,99,235,0.10), transparent 60%),
        radial-gradient(1000px 500px at -10% 20%, rgba(14,165,233,0.08), transparent 55%),
        linear-gradient(180deg, var(--bg-grad-1), var(--bg-grad-2));
    background-attachment: fixed;
}

#MainMenu, footer, header { visibility: hidden; }

/* ---- Typography ---- */
.eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 0.6rem;
}
.page-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 0.4rem;
}
.page-sub {
    font-size: 1rem;
    color: var(--text-2);
    max-width: 640px;
    line-height: 1.6;
    margin-bottom: 1.6rem;
}
.section-label {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 1.6rem 0 0.8rem;
}

/* ---- Glass card ---- */
.glass {
    background: var(--surface);
    -webkit-backdrop-filter: blur(18px);
    backdrop-filter: blur(18px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: 1.4rem;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.glass:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: rgba(37,99,235,0.25);
}
.glass .card-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.glass .card-desc {
    font-size: 0.9rem;
    color: var(--text-2);
    line-height: 1.55;
}
.glass .card-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--primary-soft), #e0f2fe);
    color: var(--primary);
    margin-bottom: 0.85rem;
}

/* ---- Metric / stat pill ---- */
.metric-pill {
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.1rem;
    box-shadow: var(--shadow-sm);
}
.metric-pill .val {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
}
.metric-pill .lbl {
    font-size: 0.78rem;
    color: var(--text-3);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ---- Confidence / risk bar ---- */
.risk-track {
    background: #eef2f7;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    position: relative;
}
.risk-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .8s cubic-bezier(.22,1,.36,1);
}
.risk-low    { background: linear-gradient(90deg,#22c55e,#4ade80); }
.risk-moderate{ background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.risk-high   { background: linear-gradient(90deg,#ef4444,#f87171); }

/* ---- Result card ---- */
.result {
    border-radius: var(--radius);
    padding: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}
.result::before {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
}
.result-ok   { background: var(--success-soft); }
.result-ok::before   { background: var(--success); }
.result-warn { background: var(--warning-soft); }
.result-warn::before { background: var(--warning); }
.result-risk { background: var(--danger-soft); }
.result-risk::before { background: var(--danger); }
.result .big {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.result-ok .big   { color: var(--success); }
.result-warn .big { color: var(--warning); }
.result-risk .big { color: var(--danger); }
.result .small { font-size: 0.88rem; color: var(--text-2); }

/* ---- Model badge ---- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.badge-ml { background: var(--primary-soft); color: var(--primary); }
.badge-ai { background: var(--warning-soft); color: var(--warning); }

/* ---- Chat ---- */
.chat-user {
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 2.5rem;
    font-size: 0.92rem;
    box-shadow: var(--shadow-sm);
}
.chat-ai {
    background: var(--surface-solid);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 2.5rem 0.4rem 0;
    font-size: 0.92rem;
    box-shadow: var(--shadow-sm);
}

/* ---- Buttons ---- */
.stButton>button,
.stFormSubmitButton>button {
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    color: #fff !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.3rem;
    font-weight: 600;
    font-family: var(--font);
    box-shadow: 0 4px 14px rgba(37,99,235,0.25);
    transition: transform .18s ease, box-shadow .18s ease;
}
.stButton>button:hover,
.stFormSubmitButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.35);
}
.stButton>button[kind="secondary"] {
    background: #fff;
    color: var(--primary) !important;
    border: 1px solid var(--border-strong);
    box-shadow: none;
}
.stDownloadButton>button {
    background: linear-gradient(135deg, var(--success), #22c55e);
    color: #fff !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-family: var(--font);
}

/* ---- Inputs ---- */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stTextArea>div>div>textarea,
.stDateInput>div>div>input,
.stSelectbox>div>div,
.stMultiSelect>div>div {
    border-radius: 12px !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--surface-solid);
    font-family: var(--font);
}
.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stDateInput label, .stTextArea label {
    font-weight: 600 !important;
    color: var(--text) !important;
    font-family: var(--font);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.78);
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.4rem 0 1rem;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-brand .logo {
    width: 40px; height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 1.1rem;
    box-shadow: 0 6px 16px rgba(37,99,235,0.35);
}
.sidebar-brand .name { font-size: 1.05rem; font-weight: 800; color: var(--text); line-height: 1.1; }
.sidebar-brand .tag { font-size: 0.72rem; color: var(--text-3); }
.sidebar-footer {
    font-size: 0.76rem;
    color: var(--text-3);
    border-top: 1px solid var(--border);
    padding-top: 0.8rem;
    margin-top: 1rem;
    line-height: 1.6;
}

/* ---- Expanders ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: rgba(255,255,255,0.6);
    box-shadow: var(--shadow-sm);
}
.streamlit-expanderHeader { font-weight: 700 !important; color: var(--text) !important; font-family: var(--font); }

/* ---- Alerts ---- */
.stAlert { border-radius: var(--radius-sm) !important; font-family: var(--font); }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.4rem 1rem;
    font-weight: 600;
    font-family: var(--font);
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }

/* ============================================================
   HUMANIZED UI TOUCHES
   ============================================================ */

/* ---- Profile avatar ---- */
.avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.05rem;
    box-shadow: 0 6px 16px rgba(37,99,235,0.30);
    flex-shrink: 0;
}
.avatar-sm {
    width: 36px;
    height: 36px;
    font-size: 0.9rem;
}

/* ---- Hero greeting banner ---- */
.hero-banner {
    background: linear-gradient(120deg, #eff6ff 0%, #e0f2fe 55%, #f0f9ff 100%);
    border: 1px solid rgba(37,99,235,0.18);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: var(--shadow-sm);
}
.hero-banner .greet {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.25;
}
.hero-banner .greet-sub {
    font-size: 0.92rem;
    color: var(--text-2);
    line-height: 1.55;
    margin-top: 0.25rem;
}

/* ---- Wellness tip chip ---- */
.wellness-tip {
    background: #fffbf1;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin: 1.4rem 0;
    font-size: 0.9rem;
    color: #78350f;
    line-height: 1.6;
    box-shadow: var(--shadow-sm);
}
.wellness-tip .tip-label {
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.25rem;
    color: #b45309;
}

/* ---- Chat timestamps ---- */
.chat-time {
    font-size: 0.68rem;
    color: var(--text-3);
    margin-bottom: 0.3rem;
}
.chat-user .chat-time { color: rgba(255,255,255,0.75); text-align: right; }
.chat-ai .chat-time { color: var(--text-3); }

/* ---- Assistant intro card ---- */
.assistant-intro {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    background: var(--surface-solid);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
}
.assistant-intro .ai-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--primary-soft), #e0f2fe);
    color: var(--primary);
    font-size: 1.2rem;
    flex-shrink: 0;
}
.assistant-intro .text { font-size: 0.9rem; color: var(--text-2); line-height: 1.55; }
.assistant-intro .text b { color: var(--text); }
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
#  HELPER: Lucide-style inline SVG icons
# ===========================================================================
def icon(name, size=18, stroke=2):
    """Return a Lucide-style inline SVG string for a given icon name."""
    paths = {
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "heart": '<path d="M19 14c1.5-1.5 3-3.2 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.8 0-3 .5-4.5 2-1.5-1.5-2.7-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4 3 5.5l7 7Z"/>',
        "droplet": '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5S12 4 12 4s-4 4.5-4 7a7 7 0 0 0 4 11z"/>',
        "ribbon": '<path d="M12 2 3 7v6c0 5 3 9 9 11 6-2 9-6 9-11V7Z"/><path d="M12 22V7"/>',
        "bot": '<rect x="3" y="8" width="18" height="12" rx="3"/><circle cx="12" cy="13" r="2"/><path d="M8 8V4h8v4"/>',
        "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
        "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
        "activity2": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
        "mail": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 5L2 7"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
        "scale": '<path d="m16 16 3-8 3 8c-.9 1.2-2 2-3 2s-2.1-.8-3-2z"/><path d="m2 16 3-8 3 8c-.9 1.2-2 2-3 2s-2.1-.8-3-2z"/><path d="M7 21h10M12 3v18M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
        "chart": '<path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
        "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
        "spark": '<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>',
        "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
        "brain": '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44A2.5 2.5 0 0 1 4 17.5v-10A2.5 2.5 0 0 1 6.5 5h3A2.5 2.5 0 0 1 12 3.5V2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44A2.5 2.5 0 0 0 20 17.5v-10A2.5 2.5 0 0 0 17.5 5h-3A2.5 2.5 0 0 0 12 3.5V2z"/>',
        "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
        "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13A4 4 0 0 1 16 11"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
        "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        "layers": '<path d="m12 2 8.5 4.5L12 11 3.5 6.5 12 2z"/><path d="m3.5 12 8.5 4.5L20.5 12"/><path d="m3.5 17.5 8.5 4.5 8.5-4.5"/>',
    }
    body = paths.get(name, paths["activity"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle">{body}</svg>'
    )


# ===========================================================================
#  HUMANIZED HELPER FUNCTIONS
# ===========================================================================
def get_initials(name):
    """Return up to 2 initials from a person's name (for the avatar)."""
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def time_greeting():
    """Return a warm greeting based on the current hour."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def avatar_html(name, size="avatar"):
    """Render a humanized profile avatar circle with the user's initials."""
    return (
        f"<div class='{size}'>{get_initials(name)}</div>"
    )


def wellness_tip():
    """Return a rotating wellness tip for a human touch."""
    import random
    tips = [
        ("💧", "Stay hydrated", "Drinking 7–8 glasses of water a day supports every system in your body."),
        ("😴", "Prioritize sleep", "Aim for 7–9 hours of quality rest — your body repairs itself while you sleep."),
        ("🚶", "Keep moving", "A brisk 30-minute daily walk can improve heart health and mood."),
        ("🥗", "Color your plate", "Eating a rainbow of fruits and vegetables delivers vital antioxidants."),
        ("🧘", "Manage stress", "Deep breathing for 5 minutes a day can lower stress hormones."),
    ]
    icon_c, title, desc = random.choice(tips)
    return icon_c, title, desc


# ===========================================================================
#  HELPER FUNCTIONS  (thin Streamlit wrappers around core.py logic)
# ===========================================================================
clean_text = core.clean_text
get_gemini_suggestions = core.get_gemini_suggestions
risk_meta = core.risk_meta


def confidence_section(probability, positive):
    """Render a professional confidence/risk indicator."""
    _, fill_cls, label = risk_meta(probability)
    pct = round(probability * 100)
    st.markdown(
        f"""
        <div class='glass' style='margin-top:1rem'>
            <div style='display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem'>
                <span style='font-weight:700;color:var(--text);font-size:0.95rem'>Risk Assessment</span>
                <span style='font-size:0.85rem;font-weight:700;color:var(--text-2)'>{label}</span>
            </div>
            <div class='risk-track'><div class='risk-fill {fill_cls}' style='width:{pct}%'></div></div>
            <div style='display:flex;justify-content:space-between;margin-top:0.5rem'>
                <span style='font-size:0.8rem;color:var(--text-3)'>Predicted probability</span>
                <span style='font-size:0.9rem;font-weight:800;color:var(--text)'>{pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(diagnosis, positive, probability):
    """Render an elegant ML result card with confidence bar."""
    if positive:
        cls, badge = "result-risk", "Detected"
    else:
        cls, badge = "result-ok", "Not Detected"
    st.markdown(
        f"""
        <div class='result {cls}'>
            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem'>
                <span class='badge badge-ml'>{icon('cpu', 13)} ML Prediction</span>
            </div>
            <div class='big'>{diagnosis}</div>
            <div class='small'>{badge} · {('Consult a healthcare professional' if positive else 'Your parameters appear within normal range')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confidence_section(probability, positive)


build_pdf = core.build_pdf  # PDF generation logic lives in core.py (unit-testable)


def email_configured():
    """Check whether email credentials are present in secrets."""
    try:
        sender = st.secrets["email"]["sender_email"]
        password = st.secrets["email"]["sender_password"]
        return bool(sender and password)
    except Exception:
        return False


def email_report(user, filename):
    """Send the report via email with graceful degradation."""
    if not email_configured():
        st.warning(
            "Email delivery isn't configured on this deployment. "
            "You can still download the report above."
        )
        return
    try:
        sender = st.secrets["email"]["sender_email"]
        password = st.secrets["email"]["sender_password"]
        msg = EmailMessage()
        msg["Subject"] = "Your Medical Report - MediPredict"
        msg["From"] = sender
        msg["To"] = user["email"]
        msg.set_content("Please find your attached medical report from MediPredict.")
        with open(filename, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=filename)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        st.success("Report sent to your email successfully.")
    except Exception as exc:
        st.error(f"Email failed to send: {exc}")


def ai_explanation_panel(disease, diagnosis, probability, parameters, user):
    """Generate a professional, personalized AI explanation for a prediction."""
    with st.spinner("Preparing your personalized health insights…"):
        prompt = f"""
You are a medical educator. Based ONLY on the following ML prediction result, write a concise,
professional, and compassionate explanation for the patient. Do NOT make a medical diagnosis.
Patient: {user['name']}, age {user['age']}, {user['gender']}.
Disease screened: {disease}.
ML prediction: {diagnosis} (predicted probability of disease: {probability*100:.1f}%).
Key parameters: { {k: str(v) for k, v in parameters.items()} }.
Structure your answer with these clear markdown headings:
1. What this result means
2. Key factors that influenced the prediction
3. Recommended next steps
4. When to consult a doctor
Keep it under 220 words, use simple language, and avoid fear-mongering.
"""
        text = get_gemini_suggestions(prompt)
    st.markdown(
        f"""
        <div class='glass' style='margin-top:1rem'>
            <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem'>
                <span class='badge badge-ai'>{icon('spark', 13)} Gemini AI Insight</span>
            </div>
            <div class='card-desc'>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_ui(disease, diagnosis, probability, parameters):
    """Expandable PDF report + email section."""
    with st.expander("Generate PDF report & email it to me"):
        user = st.session_state["user_details"]
        disease_label = disease.replace(" ", "")
        prompt = f"""
You are a medical assistant creating a personalized report for a {disease} patient.
Patient: {user['name']}, Age {user['age']}, {user['gender']}.
Diagnosis: {diagnosis}
Parameters: {parameters}
Provide a simple summary, a daily diet plan, 3 health tips, and 2 follow-up suggestions.
Use very simple, friendly English. Structure with headings.
"""
        suggestions = get_gemini_suggestions(prompt)
        filename = build_pdf(user, disease_label, diagnosis, parameters, suggestions)
        with open(filename, "rb") as f:
            st.download_button(
                "Download report (PDF)", f, file_name=filename, use_container_width=True
            )
        email_report(user, filename)


def record_history(disease, diagnosis, probability, parameters):
    """Append the current prediction to session history."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(
        {
            "disease": disease,
            "diagnosis": diagnosis,
            "probability": probability,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "parameters": dict(parameters),
        }
    )


# ===========================================================================
#  USER DETAILS FLOW
# ===========================================================================
if "user_details" not in st.session_state:
    st.session_state["user_details"] = {}
    st.session_state["details_filled"] = False
    st.session_state["history"] = []

if not st.session_state["details_filled"]:
    st.markdown(
        f"""
        <div class='hero-banner'>
            <div class='avatar'>🩺</div>
            <div>
                <div class='greet'>Welcome to MediPredict</div>
                <div class='greet-sub'>
                    A friendly, AI-powered screening companion for common health conditions.
                    Let's get to know you so we can personalize your experience.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='eyebrow'>Your data stays private & in this session</div>",
        unsafe_allow_html=True,
    )

    with st.form("user_details_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full name", placeholder="e.g. Aarav Sharma")
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        with c2:
            city = st.text_input("City", placeholder="e.g. Mumbai")
            state = st.text_input("State", placeholder="e.g. Maharashtra")
            email = st.text_input("Email", placeholder="you@example.com")

        st.markdown("")
        submitted = st.form_submit_button("Continue to dashboard", use_container_width=True)

    if submitted:
        if name.strip() and city.strip() and state.strip():
            st.session_state["user_details"] = {
                "name": name.strip(),
                "age": int(age),
                "gender": gender,
                "date": datetime.date.today().strftime("%Y-%m-%d"),
                "city": city.strip(),
                "state": state.strip(),
                "email": email.strip() or "not provided",
            }
            st.session_state["details_filled"] = True
            st.rerun()
        else:
            st.warning("Please fill in your name, city, and state to continue.")

    st.markdown(
        f"""
        <div class='glass' style='max-width:720px'>
            <div class='card-title'>{icon('shield', 18)} Privacy</div>
            <div class='card-desc'>
                Your information is used only to personalize your report, is kept in your current
                session, and is not stored on any server. Reports are generated and emailed only on request.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ===========================================================================
#  SIDEBAR
# ===========================================================================
user = st.session_state["user_details"]

with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-brand'>
            <div class='logo'>M</div>
            <div>
                <div class='name'>MediPredict</div>
                <div class='tag'>ML Disease Screening</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='glass' style='padding:0.9rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.7rem'>
            {avatar_html(user['name'], 'avatar-sm')}
            <div>
                <div style='font-weight:700;color:var(--text);font-size:0.95rem'>{user['name']}</div>
                <div style='font-size:0.8rem;color:var(--text-2)'>{user['age']} yrs · {user['gender']}</div>
                <div style='font-size:0.8rem;color:var(--text-2)'>{user['city']}, {user['state']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title="",
        options=[
            "Overview",
            "Diabetes Screening",
            "Heart Screening",
            "Breast Cancer Screening",
            "AI Health Assistant",
            "Prediction History",
            "About",
        ],
        icons=[
            "house",
            "droplet",
            "heart-pulse",
            "ribbon",
            "robot",
            "clock-history",
            "info-circle",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#2563EB", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0",
                "--hover-color": "rgba(37,99,235,0.08)",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "#ffffff",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class='sidebar-footer'>
            <b>MediPredict</b> v3.0<br>
            Predictions by scikit-learn ML<br>
            Explanations by Google Gemini<br>
            © 2024 · For education only
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
#  STATIC PAGE HEADERS
# ===========================================================================
def page_header(eyebrow_text, title, sub):
    st.markdown(f"<div class='eyebrow'>{eyebrow_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{sub}</div>", unsafe_allow_html=True)


# ===========================================================================
#  OVERVIEW (HOME)
# ===========================================================================
if selected == "Overview":
    page_header(
        f"{icon('shield',14)} ML-powered screening · {icon('spark',14)} Gemini insights",
        f"Welcome back, {user['name'].split()[0]}",
        "Screen for diabetes, heart disease, and breast cancer using validated machine-learning "
        "models. Get a clear risk assessment and a personalized, AI-written explanation.",
    )

    # Feature cards
    c1, c2, c3 = st.columns(3)
    cards = [
        ("droplet", "Diabetes Screening", "Glucose, BMI, insulin, and pedigree function assessed by a logistic-regression model."),
        ("heart", "Heart Screening", "Cholesterol, blood pressure, and ECG metrics evaluated for cardiovascular risk."),
        ("ribbon", "Breast Cancer Screening", "30 cell-nucleus features classified as benign or malignant with confidence."),
    ]
    for col, (ic, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"<div class='glass'><div class='card-icon'>{icon(ic, 20)}</div>"
                f"<div class='card-title'>{title}</div><div class='card-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-label'>Technology at a glance</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    tech = [
        ("3", "Trained models", icon("cpu")),
        ("51", "Total features", icon("layers")),
        ("LR", "Model type · Logistic Regression", icon("target")),
        ("99.9%", "Confidence reported", icon("chart")),
    ]
    for col, (val, lbl, ic) in zip([m1, m2, m3, m4], tech):
        with col:
            st.markdown(
                f"<div class='metric-pill' style='display:flex;align-items:center;gap:0.6rem'>"
                f"<span style='color:var(--primary)'>{ic}</span>"
                f"<div><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-label'>How it works</div>", unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("1", "Input parameters", "Enter your latest health readings."),
        ("2", "Model prediction", "A trained ML model classifies your risk."),
        ("3", "Risk assessment", "See a confidence score and risk level."),
        ("4", "Personal insights", "Get an AI-written explanation & next steps."),
    ]
    for col, (num, title, desc) in zip([h1, h2, h3, h4], steps):
        with col:
            st.markdown(
                f"<div class='glass'><div style='font-size:0.8rem;font-weight:800;color:var(--primary)'>{num}</div>"
                f"<div class='card-title' style='font-size:0.95rem;margin:0.2rem 0'>{title}</div>"
                f"<div class='card-desc' style='font-size:0.82rem'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class='glass' style='margin-top:1.6rem;border-left:4px solid var(--warning)'>
            <div class='card-title'>{icon('alert', 18)} Important disclaimer</div>
            <div class='card-desc'>
                MediPredict is for <b>educational and research purposes only</b>. It is not a medical device
                and does not replace professional diagnosis. Always consult a qualified healthcare provider
                for any health concerns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
#  GENERIC SCREENING PAGE BUILDER
# ===========================================================================
def screening_page(disease_key, model_key, title, fields, help_text, sample, autofill_label):
    """Build a consistent, professional screening form for any disease."""
    page_header(
        f"{icon('shield',14)} ML screening · Logistic Regression",
        title,
        help_text,
    )

    # Initialize inputs
    s_state_key = f"{model_key}_inputs"
    if s_state_key not in st.session_state:
        st.session_state[s_state_key] = {f: 0.0 for f in fields}

    tb1, tb2 = st.columns([1, 3])
    with tb1:
        if st.button(f"{icon('refresh', 14)} Autofill sample", use_container_width=True):
            st.session_state[s_state_key] = sample.copy()
    with tb2:
        st.caption("Sample values demonstrate the model with realistic clinical data.")

    c1, c2 = st.columns(2)
    half = len(fields) // 2 + len(fields) % 2
    with c1:
        for f in fields[:half]:
            st.session_state[s_state_key][f] = st.number_input(
                f, value=float(st.session_state[s_state_key][f]), key=f"{model_key}_{f}",
            )
    with c2:
        for f in fields[half:]:
            st.session_state[s_state_key][f] = st.number_input(
                f, value=float(st.session_state[s_state_key][f]), key=f"{model_key}_{f}",
            )

    if st.button(f"{icon('cpu', 16)} Run {model_key} model", use_container_width=True):
        model = MODELS.get(model_key)
        if model is None:
            st.error("Model not loaded. Please check the .pkl file exists.")
        else:
            vals = list(st.session_state[s_state_key].values())
            proba = model.predict_proba([vals])[0]
            result = int(model.predict([vals])[0])
            positive = result == 1
            probability = float(proba[1]) if positive else float(proba[0])

            disease_label = model_key.replace("_", " ").title()
            diagnosis = (
                f"{disease_label} Detected" if positive else f"No {disease_label} Detected"
            )

            result_card(diagnosis, positive, probability)
            ai_explanation_panel(disease_key, diagnosis, probability, st.session_state[s_state_key], user)
            record_history(disease_key, diagnosis, probability, st.session_state[s_state_key])
            report_ui(disease_key, diagnosis, probability, st.session_state[s_state_key])


# ===========================================================================
#  DIABETES
# ===========================================================================
DIABETES_FIELDS = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree Function", "Age",
]
DIABETES_SAMPLE = {
    "Pregnancies": 6, "Glucose": 148, "Blood Pressure": 72, "Skin Thickness": 35,
    "Insulin": 0, "BMI": 33.6, "Diabetes Pedigree Function": 0.627, "Age": user["age"],
}
DIABETES_HELP = (
    "Provide your latest glucose, blood-pressure, and BMI readings. The model evaluates "
    "insulin and family history to estimate diabetes risk."
)

if selected == "Diabetes Screening":
    screening_page(
        "Diabetes", "diabetes", "Diabetes Screening",
        DIABETES_FIELDS, DIABETES_HELP, DIABETES_SAMPLE, "Diabetes",
    )

    with st.expander("Quick BMI calculator"):
        wc, hc = st.columns(2)
        with wc:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.5)
        with hc:
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=0.5)
        if st.button("Calculate BMI"):
            if height > 0:
                bmi = weight / ((height / 100) ** 2)
                st.session_state.diabetes_inputs["BMI"] = round(bmi, 2)
                st.success(f"Your BMI is **{bmi:.2f}** — auto-filled into the form.")
                if bmi < 18.5:
                    st.caption("Category: Underweight")
                elif bmi < 25:
                    st.caption("Category: Normal weight")
                elif bmi < 30:
                    st.caption("Category: Overweight")
                else:
                    st.caption("Category: Obese")


# ===========================================================================
#  HEART
# ===========================================================================
HEART_FIELDS = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "FBS > 120", "Rest ECG",
    "Max Heart Rate", "Exercise Angina", "Oldpeak", "Slope", "CA", "Thal",
]
HEART_SAMPLE = {
    "Age": user["age"], "Sex": 1, "Chest Pain Type": 3, "Resting BP": 130,
    "Cholesterol": 250, "FBS > 120": 0, "Rest ECG": 1, "Max Heart Rate": 150,
    "Exercise Angina": 0, "Oldpeak": 1.0, "Slope": 2, "CA": 0, "Thal": 2,
}
HEART_HELP = (
    "Enter your resting blood pressure, cholesterol, and exercise-related metrics. "
    "The model weighs ECG and angina signals to estimate cardiovascular risk."
)

if selected == "Heart Screening":
    screening_page(
        "Heart Disease", "heart", "Heart Screening",
        HEART_FIELDS, HEART_HELP, HEART_SAMPLE, "Heart",
    )


# ===========================================================================
#  BREAST CANCER
# ===========================================================================
BC_FIELDS = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension",
]
BC_SAMPLE = {
    f: v for f, v in zip(
        BC_FIELDS,
        [14.0, 20.0, 90.0, 600.0, 0.1, 0.2, 0.2, 0.1, 0.2, 0.06,
         1.0, 0.9, 6.0, 100.0, 0.007, 0.02, 0.03, 0.01, 0.02, 0.003,
         15.0, 25.0, 100.0, 800.0, 0.14, 0.3, 0.4, 0.2, 0.3, 0.1],
    )
}
BC_HELP = (
    "These 30 cell-nucleus features are derived from a fine-needle aspirate image. "
    "The model classifies the tumor as benign or malignant with a confidence score."
)

if selected == "Breast Cancer Screening":
    screening_page(
        "Breast Cancer", "breast", "Breast Cancer Screening",
        BC_FIELDS, BC_HELP, BC_SAMPLE, "Breast",
    )


# ===========================================================================
#  AI HEALTH ASSISTANT
# ===========================================================================
if selected == "AI Health Assistant":
    page_header(
        f"{icon('spark',14)} Gemini AI",
        "Health Assistant",
        "Ask health-related questions in natural language and get clear, educational answers "
        "from Google Gemini.",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    quick = ["What does high BMI mean?", "Healthy diet for diabetes", "Explain cholesterol levels"]
    st.markdown("**Suggested topics**")
    qcols = st.columns(len(quick))
    for col, q in zip(qcols, quick):
        with col:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                with st.spinner("Thinking…"):
                    st.session_state.chat_history.append(("ai", get_gemini_suggestions(q)))
                st.rerun()

    user_input = st.text_area(
        "Your question", placeholder="e.g. Explain the importance of HbA1c…", height=90
    )
    b1, b2 = st.columns([1, 3])
    with b1:
        ask = st.button("Ask assistant", use_container_width=True)
    with b2:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if ask:
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking…"):
                st.session_state.chat_history.append(("ai", get_gemini_suggestions(user_input)))
                st.rerun()
        else:
            st.warning("Please enter a question first.")

    st.markdown("")
    if not st.session_state.chat_history:
        st.info("Start a conversation by asking a health-related question above.")
    for role, msg in reversed(st.session_state.chat_history):
        cls = "chat-user" if role == "user" else "chat-ai"
        st.markdown(f"<div class='{cls}'>{msg}</div>", unsafe_allow_html=True)


# ===========================================================================
#  PREDICTION HISTORY
# ===========================================================================
if selected == "Prediction History":
    page_header(
        f"{icon('chart',14)} Session log",
        "Prediction History",
        "A record of every screening you've run in this session.",
    )

    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Run a screening to see results here.")
    else:
        rows = []
        for item in reversed(history):
            rows.append(
                {
                    "Date": item["date"],
                    "Screening": item["disease"],
                    "Result": item["diagnosis"],
                    "Probability": f"{item['probability']*100:.1f}%",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if history and st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()


# ===========================================================================
#  ABOUT
# ===========================================================================
if selected == "About":
    page_header(
        f"{icon('info',14)} About this project",
        "About MediPredict",
        "A production-quality machine-learning screening platform built for education and research.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class='glass'>
                <div class='card-icon'>{icon('target', 20)}</div>
                <div class='card-title'>What it does</div>
                <div class='card-desc'>
                    MediPredict applies three validated <b>scikit-learn logistic-regression models</b> to
                    screen for diabetes, heart disease, and breast cancer. Each result includes a
                    confidence score and risk level, plus a personalized explanation written by
                    <b>Google Gemini</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='glass'>
                <div class='card-icon'>{icon('layers', 20)}</div>
                <div class='card-title'>Technology stack</div>
                <div class='card-desc'>
                    <b>Python</b> · <b>Streamlit</b> · <b>Scikit-learn</b><br>
                    <b>Google Gemini</b> · <b>FPDF</b> reports · <b>SMTP</b> email<br>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-label'>Models</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    models_info = [
        ("droplet", "Diabetes", "8 features · Pregnancies, Glucose, BP, BMI, Insulin, Pedigree, Age"),
        ("heart", "Heart Disease", "13 features · BP, Cholesterol, Max HR, ECG, Angina, Slope, CA, Thal"),
        ("ribbon", "Breast Cancer", "30 features · Cell nucleus size, shape, texture & boundary"),
    ]
    for col, (ic, title, desc) in zip([d1, d2, d3], models_info):
        with col:
            st.markdown(
                f"<div class='glass'><div class='card-icon'>{icon(ic, 20)}</div>"
                f"<div class='card-title'>{title}</div><div class='card-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class='glass' style='margin-top:1.6rem;border-left:4px solid var(--warning)'>
            <div class='card-title'>{icon('alert', 18)} Disclaimer</div>
            <div class='card-desc'>
                This application is for <b>educational and research purposes only</b>. It is not a medical
                device and should not replace professional diagnosis. Always consult a qualified healthcare
                provider for any medical questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
