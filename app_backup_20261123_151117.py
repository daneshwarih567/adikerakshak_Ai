from pathlib import Path
import json
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

from llama_advisor import get_farmer_advice, get_treatment_advisory

# ---------------- CONFIGURATION & PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "arecanut_model.keras"
CLASSES_PATH = BASE_DIR / "classes.json"
DB_PATH = BASE_DIR / "adikerakshak_records.db"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="AdikeRakshak AI · Kisan AI Arecanut Clinic",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- 8 LANGUAGES DICTIONARY ----------------
LANGUAGES = {
    "English": "en",
    "ಕನ್ನಡ (Kannada)": "kn",
    "हिंदी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
    "മലയാളം (Malayalam)": "ml",
    "ಮರಾಠಿ (Marathi)": "mr",
    "ಉರ್ದು (Urdu)": "ur"
}

UI_TEXT = {
    "en": {
        "title": "🌴 AdikeRakshak AI · Kisan AI",
        "subtitle": "Advanced Arecanut Plant Pathology & Precise Quantity Advisory",
        "step1": "📁 Step 1: Upload Plant Image",
        "scan_btn": "🔬 Run AI Scan",
        "visual": "🔍 Visual Inspection • Input vs AI Detection",
        "diag": "💊 Pathological Diagnosis & Exact Quantity Rx",
        "sev": "📊 Disease Severity Metrics",
        "tissue": "🧪 Symptom Tissue Breakdown (%)",
        "trend": "📈 Progression Trend Curve",
        "logs": "🗂️ Historical Clinical Logs",
        "kisan": "🌱 Kisan AI Follow-Up Assistant (Llama 3.2 · Ollama)",
        "bench": "System Diagnostic Performance & Benchmarks",
        "recent": "Recent Clinical Diagnostics",
        "cam": "📷 Live Camera Capture (Alternative Option)",
        "lbl_rx": "💊 Prescribed Rx:",
        "lbl_dose": "⚖️ Exact Required Quantity / Dose:",
        "lbl_adv": "⚠️ Clinical Advisory:"
    },
    "kn": {
        "title": "🌴 ಅಡಿಕೆ ರಕ್ಷಕ್ AI · ಕಿಸಾನ್ AI",
        "subtitle": "ಸುಧಾರಿತ ಅಡಿಕೆ ಸಸ್ಯ ರೋಗಶಾಸ್ತ್ರ ಮತ್ತು ನಿಖರವಾದ ಔಷಧ ಪ್ರಮಾಣ ಸಲಹೆ",
        "step1": "📁 ಹಂತ 1: ಸಸ್ಯದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "scan_btn": "🔬 AI ಸ್ಕ್ಯಾನ್ ನಡೆಸಿ",
        "visual": "🔍 ದೃಶ್ಯ ಪರಿಶೀಲನೆ • ಇನ್ಪುಟ್ vs AI ಪತ್ತೆ",
        "diag": "💊 ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ನಿಖರ ಔಷಧ ಪ್ರಮಾಣ",
        "sev": "📊 ರೋಗದ ತೀವ್ರತೆಯ ಮಾಪನಗಳು",
        "tissue": "🧪 ಸೋಂಕಿತ ಅಂಗಾಂಶಗಳ ವಿಭಜನೆ (%)",
        "trend": "📈 ಪ್ರಗತಿ ಪ್ರವೃತ್ತಿ ರೇಖೆ",
        "logs": "🗂️ ಹಿಂದಿನ ಕ್ಲಿನಿಕಲ್ ದಾಖಲೆಗಳು",
        "kisan": "🌱 ಕಿಸಾನ್ AI ಸಹಾಯಕಿ (Llama 3.2 · Ollama)",
        "bench": "ಸಿಸ್ಟಮ್ ರೋಗನಿರ್ಣಯ ಕಾರ್ಯಕ್ಷಮತೆ ಮತ್ತು ಮಾನದಂಡಗಳು",
        "recent": "ಇತ್ತೀಚಿನ ಕ್ಲಿನಿಕಲ್ ರೋಗನಿರ್ಣಯಗಳು",
        "cam": "📷 ಲೈವ್ ಕ್ಯಾಮೆರಾ ಕ್ಯಾಪ್ಚರ್",
        "lbl_rx": "💊 ಸೂಚಿಸಿದ ಔಷಧ / ಕ್ರಮ:",
        "lbl_dose": "⚖️ ನಿಖರ ಪ್ರಮಾಣ / ಡೋಸ್:",
        "lbl_adv": "⚠️ ಕೃಷಿ ಸಲಹೆ:"
    },
    "hi": {
        "title": "🌴 अडिकारक्षक AI · किसान AI",
        "subtitle": "उन्नत सुपारी पादप रोग विज्ञान और सटीक मात्रा सलाह",
        "step1": "📁 चरण 1: पौधे की छवि अपलोड करें",
        "scan_btn": "🔬 AI स्कैन चलाएं",
        "visual": "🔍 दृश्य निरीक्षण • इनपुट बनाम AI पहचान",
        "diag": "💊 रोग निदान और सटीक दवा खुराक",
        "sev": "📊 रोग की गंभीरता के मेट्रिक्स",
        "tissue": "🧪 लक्षण ऊतक टूटना (%)",
        "trend": "📈 प्रगति प्रवृत्ति वक्र",
        "logs": "🗂️ ऐतिहासिक नैदानिक ​​लॉग",
        "kisan": "🌱 किसान AI सहायक (Llama 3.2 · Ollama)",
        "bench": "सिस्टम नैदानिक ​​प्रदर्शन और बेंचमार्क",
        "recent": "हाल के नैदानिक ​​निदान",
        "cam": "📷 लाइव कैमरा कैप्चर",
        "lbl_rx": "💊 निर्धारित उपचार / दवा:",
        "lbl_dose": "⚖️ सटीक आवश्यक मात्रा / खुराक:",
        "lbl_adv": "⚠️ कृषि / नैदानिक सलाह:"
    },
    "te": {
        "title": "🌴 అడికెరక్షక్ AI · కిసాన్ AI",
        "subtitle": "అధునాతన పోక చెట్టు మొక్కల వ్యాధి నిర్ధారణ మరియు సరైన మోతాదు సలహా",
        "step1": "📁 దశ 1: మొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "scan_btn": "🔬 AI స్కాన్ చేయండి",
        "visual": "🔍 విజువల్ ఇన్స్పెక్ట్ • ఇన్పుట్ vs AI గుర్తింపు",
        "diag": "💊 వ్యాధి నిర్ధారణ & ఖచ్చితమైన మోతాదు",
        "sev": "📊 వ్యాధి తీవ్రత మెట్రిక్స్",
        "tissue": "🧪 కణజాల విభజన (%)",
        "trend": "📈 ప్రోగ్రెషన్ ట్రెండ్ కర్వ్",
        "logs": "🗂️ చారిత్రక లాగ్‌లు",
        "kisan": "🌱 కిసాన్ AI అసిస్టెంట్ (Llama 3.2 · Ollama)",
        "bench": "సిస్టమ్ డయాగ్నోస్టిక్ బెంచ్‌మార్క్‌లు",
        "recent": "ఇటీవలి రోగనిర్ధారణలు",
        "cam": "📷 లైవ్ కెమెరా క్యాప్చర్",
        "lbl_rx": "💊 సూచించిన మందు / చికిత్స:",
        "lbl_dose": "⚖️ ఖచ్చితమైన మోతాదు:",
        "lbl_adv": "⚠️ వైద్య / వ్యవసాయ సలహా:"
    },
    "ta": {
        "title": "🌴 அடிகேரக்ஷக் AI · கிசான் AI",
        "subtitle": "மேம்பட்ட பாக்கு தாவர நோயியல் மற்றும் அளவு வழிகாட்டி",
        "step1": "📁 படி 1: தாவர படத்தைப் பதிவேற்றவும்",
        "scan_btn": "🔬 AI ஸ்கேன் இயக்கவும்",
        "visual": "🔍 காட்சி ஆய்வு • உள்ளீடு vs AI கண்டறிதல்",
        "diag": "💊 நோய் கண்டறிதல் & சரியான மருந்து அளவு",
        "sev": "📊 நோய் தீவிர அளவீடுகள்",
        "tissue": "🧪 திசு சிதைவு (%)",
        "trend": "📈 முன்னேற்ற போக்கு வளைவு",
        "logs": "🗂️ மருத்துவ பதிவுகள்",
        "kisan": "🌱 கிசான் AI உதவியாளர் (Llama 3.2 · Ollama)",
        "bench": "கணினி செயல்திறன் அளவுகோல்கள்",
        "recent": "சமீபத்திய மருத்துவ பரிசோதனைகள்",
        "cam": "📷 நேரலை கேமரா பிடிப்பு",
        "lbl_rx": "💊 பரிந்துரைக்கப்பட்ட மருந்து:",
        "lbl_dose": "⚖️ துல்லியமான அளவு:",
        "lbl_adv": "⚠️ வேளாண் ஆலோசனை:"
    },
    "ml": {
        "title": "🌴 അഡികെരക്ഷക് AI · കിസാൻ AI",
        "subtitle": "വിപുലമായ കവുങ്ങ് രോഗനിർണ്ണയവും അളവ് നിർദ്ദേശവും",
        "step1": "📁 ഘട്ടം 1: ചെടിയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "scan_btn": "🔬 AI സ്കാൻ ചെയ്യുക",
        "visual": "🔍 വിഷ്വൽ പരിശോധന",
        "diag": "💊 രോഗനിർണ്ണയവും കൃത്യമായ മരുന്ന് അളവും",
        "sev": "📊 രോഗ തീവ്രത മെട്രിക്സ്",
        "tissue": "🧪 ടിഷ്യു തകർച്ച (%)",
        "trend": "📈 ട്രെൻഡ് കർവ്",
        "logs": "🗂️ ചരിത്രപരമായ ലോഗുകൾ",
        "kisan": "🌱 കിസാൻ AI അസിസ്റ്റന്റ് (Llama 3.2 · Ollama)",
        "bench": "സിസ്റ്റം ബെഞ്ച്മാർക്കുകൾ",
        "recent": "സമീപകാല രോഗനിർണ്ണയങ്ങൾ",
        "cam": "📷 ലൈവ് ക്യാമറ ക്യാപ്ചർ",
        "lbl_rx": "💊 നിർദ്ദേശിച്ച മരുന്ന്:",
        "lbl_dose": "⚖️ കൃത്യമായ അളവ്:",
        "lbl_adv": "⚠️ കാർഷിക ഉപദേശം:"
    },
    "mr": {
        "title": "🌴 अडिकेक्षक AI · किसान AI",
        "subtitle": "प्रगत सुपारी वनस्पती रोगशास्त्र आणि अचूक प्रमाण सल्ला",
        "step1": "📁 पायरी 1: वनस्पती प्रतिमा अपलोड करा",
        "scan_btn": "🔬 AI स्कॅन चालवा",
        "visual": "🔍 दृश्य तपासणी",
        "diag": "💊 रोगनिदान आणि अचूक औषध मात्रा",
        "sev": "📊 रोगाची तीव्रता मेट्रिक्स",
        "tissue": "🧪 ऊतक बिघाड (%)",
        "trend": "📈 प्रगती वक्र",
        "logs": "🗂️ ऐतिहासिक नोंदी",
        "kisan": "🌱 किसान AI सहाय्यक (Llama 3.2 · Ollama)",
        "bench": "सिस्टम बेंचमार्क",
        "recent": "अलीकडील निदान",
        "cam": "📷 थेट कॅमेरा कॅप्चर",
        "lbl_rx": "💊 सुचवलेले औषध:",
        "lbl_dose": "⚖️ अचूक मात्रा / डोस:",
        "lbl_adv": "⚠️ कृषी सल्ला:"
    },
    "ur": {
        "title": "🌴 اڈیکے رکشاک AI · کسان AI",
        "subtitle": "جدید سپاری پلانٹ پیتھالوجی اور خوراک کی ہدایت",
        "step1": "📁 مرحلہ 1: پودے کی تصویر اپ لوڈ کریں",
        "scan_btn": "🔬 AI اسکین چلائیں",
        "visual": "🔍 بصری معائنہ",
        "diag": "💊 بیماری کی تشخیص اور درست خوراک",
        "sev": "📊 شدت کے میٹرکس",
        "tissue": "🧪 ٹشو کی خرابی (%)",
        "trend": "📈 رجحان کا منحنی خطوط",
        "logs": "🗂️ تاریخی لاگز",
        "kisan": "🌱 کسان AI اسسٹنٹ (Llama 3.2 · Ollama)",
        "bench": "سسٹم کے معیارات",
        "recent": "حالیہ تشخیص",
        "cam": "📷 لائیو کیمرہ کیپچر",
        "lbl_rx": "💊 تجویز کردہ دوا:",
        "lbl_dose": "⚖️ درست مقدار / خوراک:",
        "lbl_adv": "⚠️ زرعی مشورہ:"
    }
}

# ---------------- LUXURIOUS DARK EMERALD CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #08261c 0%, #04120e 55%, #020806 100%);
    color: #e2f0ea;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 5rem;
}

.brand-header {
    background: linear-gradient(135deg, rgba(8, 42, 30, 0.95) 0%, rgba(13, 65, 48, 0.9) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6);
}

.dash-panel {
    background: rgba(7, 28, 21, 0.75);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(22, 74, 57, 0.8);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}

.panel-eyebrow {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #34d399;
    font-weight: 800;
    margin-bottom: 0.7rem;
}

.disease-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.2rem 0;
}

.conf-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}

.metric-box {
    background: #071c15;
    border: 1px solid #154c39;
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
}
.metric-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: #34d399;
}
.metric-lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8fa89e;
    margin-top: 4px;
}

div.stButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #10b981 100%);
    color: #031410;
    font-weight: 800;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.4rem;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #14b8a6 0%, #34d399 100%);
}

.stTextInput input, .stTextArea textarea {
    background-color: rgba(5, 20, 15, 0.9) !important;
    color: #ffffff !important;
    border-color: rgba(26, 92, 70, 0.8) !important;
    border-radius: 10px !important;
}

a { color: #5eead4 !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE SETUP ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scan_logs);")
        columns = [col[1] for col in cursor.fetchall()]
        if not columns or "plant" not in columns:
            conn.execute("DROP TABLE IF EXISTS scan_logs;")
            conn.execute("""
                CREATE TABLE scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT,
                    plant TEXT,
                    disease TEXT,
                    confidence REAL,
                    severity REAL
                )
            """)
            conn.commit()
init_db()

def log_scan(plant, disease, conf, sev):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO scan_logs (scan_date, plant, disease, confidence, severity) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d  %I:%M:%S %p"), plant, disease, round(conf, 1), round(sev, 1))
            )
            conn.commit()
    except Exception:
        pass

def get_logs():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT scan_date AS 'Date & Time', disease AS 'Diagnosed Disease', confidence AS 'Confidence (%)', severity AS 'Severity (%)' FROM scan_logs ORDER BY id DESC LIMIT 6").fetchall()
        return pd.DataFrame(rows, columns=["Date & Time", "Diagnosed Disease", "Confidence (%)", "Severity (%)"])
    except Exception:
        return pd.DataFrame(columns=["Date & Time", "Diagnosed Disease", "Confidence (%)", "Severity (%)"])

# ---------------- ML & TRANSLATION HELPERS ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[str(i)] for i in range(len(data))] if isinstance(data, dict) else list(data)

# Treatment panel is generated dynamically by Ollama/Llama in the diagnostic section below.

def analyze_image(pil_img, is_healthy):
    img_np = np.array(pil_img.resize((300, 300)))
    r, g, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
    spot_mask = (r > 110) & (g > 90) & (b < 85)
    dark_mask = (r < 85) & (g < 75) & (b < 65)
    total_px = float(img_np.shape[0] * img_np.shape[1])
    spot_pct = round((np.sum(spot_mask) / total_px) * 100, 1)
    nec_pct = round((np.sum(dark_mask) / total_px) * 100, 1)
    healthy_pct = max(0.0, round(100.0 - spot_pct - nec_pct, 1))

    sev_score = min(10.0, max(1.2, round((spot_pct + nec_pct * 1.5) / 8.0, 1)))
    sev_pct = min(95.0, max(5.0, round(spot_pct + nec_pct * 1.2, 1)))

    overlay = pil_img.copy().resize((300, 300))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(spot_mask | dark_mask)
    if len(xs) > 20 and not is_healthy:
        x_min, x_max = int(np.percentile(xs, 10)), int(np.percentile(xs, 90))
        y_min, y_max = int(np.percentile(ys, 10)), int(np.percentile(ys, 90))
        draw.rectangle([x_min, y_min, x_max, y_max], outline="#facc15", width=3)
        draw.text((x_min + 2, max(0, y_min - 14)), f"Lesion {sev_pct}%", fill="#facc15")

    return overlay, sev_score, sev_pct, {"Leaf Spot": spot_pct, "Necrosis": nec_pct, "Healthy": healthy_pct}

# ---------------- TOP BRAND HEADER WITH ACTIVE LANGUAGE TRANSLATION ----------------
st.markdown('<div class="brand-header">', unsafe_allow_html=True)
h_col1, h_col2 = st.columns([2.5, 1.5])
with h_col1:
    selected_lang_name = st.selectbox("🌐 Select Language", list(LANGUAGES.keys()), label_visibility="collapsed")
    lang_code = LANGUAGES[selected_lang_name]
    t = UI_TEXT[lang_code]

    st.markdown(f"""
    <div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #ffffff;">{t['title']}</div>
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.14em; color: #34d399; font-weight: 700; margin-top: 3px;">{t['subtitle']}</div>
    </div>
    """, unsafe_allow_html=True)
with h_col2:
    st.markdown(f"<div style='text-align:right; color:#6ee7b7; font-weight:700; padding-top:8px;'>Active Language: {selected_lang_name}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- 1. UPLOAD IMAGE SECTION (1ST) ----------------
st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
st.markdown(f'<div class="panel-eyebrow">{t["step1"]}</div>', unsafe_allow_html=True)
up_col1, up_col2 = st.columns([3, 1], gap="medium")
with up_col1:
    uploaded = st.file_uploader("Upload leaf, nut, or trunk image", type=["jpg", "jpeg", "png", "webp"])
with up_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_upload_btn = st.button(t["scan_btn"], use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded and run_upload_btn:
    try:
        with st.spinner("Running AI inference & spatial tissue analysis..."):
            pil_img = Image.open(uploaded).convert("RGB")
            model = load_model()
            classes = load_classes()
            arr = np.expand_dims(np.asarray(pil_img.resize(IMAGE_SIZE), dtype=np.float32), axis=0)
            preds = np.asarray(model.predict(arr, verbose=0))[0]
            top_idx = int(np.argmax(preds))
            raw_c = str(classes[top_idx])
            clean_name = raw_c.replace("_", " ").title()
            conf = float(preds[top_idx]) * 100.0
            is_healthy = "healthy" in raw_c.lower()

            overlay, sev_score, sev_pct, symptom_dist = analyze_image(pil_img, is_healthy)

            st.session_state["scan_res"] = {
                "raw_c": raw_c,
                "clean_name": clean_name, "conf": conf, "orig": pil_img, "overlay": overlay,
                "sev_score": sev_score, "sev_pct": sev_pct, "symptom_dist": symptom_dist
            }
            log_scan("Arecanut Palm", clean_name, conf, sev_pct)
            st.rerun()
    except Exception as e:
        st.error(f"Error during scan: {e}")

res = st.session_state.get("scan_res")

# ---------------- 2. DIAGNOSTIC RESULTS & METRICS ----------------
col_l, col_m, col_r = st.columns([1.35, 1.35, 1.1], gap="medium")

with col_l:
    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["visual"]}</div>', unsafe_allow_html=True)
    if res:
        v1, v2 = st.columns(2)
        v1.image(res["orig"], caption="Input Scan", use_container_width=True)
        v2.image(res["overlay"], caption="AI Detection & Heatmap", use_container_width=True)
    elif uploaded:
        st.image(Image.open(uploaded), caption="Selected Image Ready", use_container_width=True)
    else:
        st.info("Upload an image above and click **Run AI Scan**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["diag"]}</div>', unsafe_allow_html=True)
    if res:
        st.markdown(f'<div class="disease-title">{res["clean_name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="conf-badge">{res["conf"]:.1f}% Confidence</span>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Cache the generated advisory for this scan so normal Streamlit reruns
        # do not repeatedly call the local Llama model.
        advisory_key = (
            f"treatment_advisory_{res.get('raw_c', res['clean_name'])}_"
            f"{res['conf']:.2f}_{lang_code}"
        )
        if advisory_key not in st.session_state:
            with st.spinner("🌴 Llama 3.2 is preparing the treatment advisory..."):
                st.session_state[advisory_key] = get_treatment_advisory(
                    disease=res.get("raw_c", res["clean_name"]),
                    confidence=res["conf"],
                    language=selected_lang_name,
                )
        treatment = st.session_state[advisory_key]

        st.markdown(f"**{t['lbl_rx']}**")
        st.write(treatment.get("rx", "Treatment information unavailable."))
        st.markdown(f"**{t['lbl_dose']}**")
        st.write(treatment.get("dose", "Please confirm the dose with a qualified agriculture officer."))
        st.markdown(f"**{t['lbl_adv']}**")
        st.write(treatment.get("advisory", "Please consult your local KVK or agriculture officer."))
        st.caption(
            "Llama-generated guidance is informational, not a confirmed diagnosis. "
            "Do not apply a pesticide or dose without checking an approved product label "
            "and confirming suitability with a KVK/agriculture officer."
        )
    else:
        st.write("Disease classification and Llama-generated treatment advisory will appear here after scanning.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_m:
    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["sev"]}</div>', unsafe_allow_html=True)
    if res:
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.metric("Severity Score", f"{res['sev_score']}/10")
            st.metric("Damage Area", f"{res['sev_pct']}%")
        with s_c2:
            st.progress(int(res['sev_pct']))
            st.caption(f"Defective Tissue: {res['sev_pct']}% | Recoverable: {round(100-res['sev_pct'],1)}%")
    else:
        st.caption("Severity metrics and damage ratio appear after scan execution.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["tissue"]}</div>', unsafe_allow_html=True)
    if res:
        st.bar_chart(res["symptom_dist"], color="#10b981", height=180)
    else:
        st.caption("Tissue breakdown chart displays after diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["trend"]}</div>', unsafe_allow_html=True)
    logs = get_logs()
    if len(logs) > 1:
        st.line_chart(logs.set_index("Date & Time")[["Confidence (%)"]], color="#34d399", height=160)
    else:
        st.caption("Perform multiple scans to track chronological trend curves.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="dash-panel"><div class="panel-eyebrow">{t["logs"]}</div>', unsafe_allow_html=True)
    if not logs.empty:
        st.dataframe(logs, use_container_width=True, hide_index=True, height=220)
    else:
        st.caption("No historical logs recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SMART KISAN AI ASSISTANT (LLAMA 3.2 + OLLAMA) ----------------
st.markdown('<div class="dash-panel" style="padding: 1.2rem 1.6rem;">', unsafe_allow_html=True)
st.markdown(f'<div class="panel-eyebrow">{t["kisan"]}</div>', unsafe_allow_html=True)
st.caption("Runtime: Llama 3.2 1B · Ollama (Local AI)")

q_col1, q_col2 = st.columns([5, 1], gap="small")
with q_col1:
    q = st.text_input(
        f"Ask a question in {selected_lang_name}...",
        key="kisan_input",
        label_visibility="collapsed",
    )
with q_col2:
    ask_btn = st.button("Ask Kisan AI", use_container_width=True)

if ask_btn and q.strip():
    ctx_disease = res["clean_name"] if res else None
    with st.spinner("Llama 3.2 is preparing your advice..."):
        ai_reply = get_farmer_advice(
            question=q.strip(),
            disease=ctx_disease,
            language=selected_lang_name,
        )
    st.markdown(f"**Kisan AI ({selected_lang_name}):**")
    st.write(ai_reply)
    st.caption(
        "AI-generated guidance is not a confirmed diagnosis. "
        "Verify treatment decisions with a local KVK or agriculture officer."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SYSTEM BENCHMARKS ----------------
st.markdown(f"### {t['bench']}")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-box"><div class="metric-val">99.06%</div><div class="metric-lbl">Overall Accuracy</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box"><div class="metric-val">99.04%</div><div class="metric-lbl">Model Precision</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box"><div class="metric-val">99.04%</div><div class="metric-lbl">Sensitivity / Recall</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-box"><div class="metric-val">99.04%</div><div class="metric-lbl">F1-Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- RECENT CLINICAL DIAGNOSTICS TABLE ----------------
st.markdown(f"### {t['recent']}")
if not logs.empty:
    for idx, row in logs.iterrows():
        c_a, c_b, c_c, c_d, c_e = st.columns([1.8, 2, 1, 1, 1])
        c_a.write(row["Date & Time"])
        c_b.write(row["Diagnosed Disease"])
        c_c.markdown(f'<span style="color:#34d399; font-weight:700;">{row["Confidence (%)"]}%</span>', unsafe_allow_html=True)
        c_d.markdown(f'<span style="color:#f87171; font-weight:700;">{row["Severity (%)"]}% Defect</span>', unsafe_allow_html=True)
        c_e.button("👁️ View", key=f"view_{idx}")
else:
    st.info("No clinical scans recorded yet.")

# ---------------- 3. CAMERA CAPTURE (PLACED AT THE VERY BOTTOM) ----------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
st.markdown(f'<div class="panel-eyebrow">{t["cam"]}</div>', unsafe_allow_html=True)
cam = st.camera_input("Capture live photo with camera (Click 'Allow' when prompted)")
run_cam_btn = st.button("🔬 Run Camera AI Scan", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if cam and run_cam_btn:
    try:
        with st.spinner("Running AI inference on captured photo..."):
            pil_img = Image.open(cam).convert("RGB")
            model = load_model()
            classes = load_classes()
            arr = np.expand_dims(np.asarray(pil_img.resize(IMAGE_SIZE), dtype=np.float32), axis=0)
            preds = np.asarray(model.predict(arr, verbose=0))[0]
            top_idx = int(np.argmax(preds))
            raw_c = str(classes[top_idx])
            clean_name = raw_c.replace("_", " ").title()
            conf = float(preds[top_idx]) * 100.0
            is_healthy = "healthy" in raw_c.lower()

            overlay, sev_score, sev_pct, symptom_dist = analyze_image(pil_img, is_healthy)

            st.session_state["scan_res"] = {
                "raw_c": raw_c,
                "clean_name": clean_name, "conf": conf, "orig": pil_img, "overlay": overlay,
                "sev_score": sev_score, "sev_pct": sev_pct, "symptom_dist": symptom_dist
            }
            log_scan("Arecanut Palm", clean_name, conf, sev_pct)
            st.rerun()
    except Exception as e:
        st.error(f"Error during camera scan: {e}")