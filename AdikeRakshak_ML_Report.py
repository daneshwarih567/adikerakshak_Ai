
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from treatment_data_new import TREATMENT_DATA

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "arecanut_model.keras"
CLASSES_PATH = BASE_DIR / "classes.json"

st.set_page_config(
    page_title="AdikeRakshak AI",
    page_icon="🌴",
    layout="centered"
)

# ---------------- EIGHT LANGUAGES ----------------

LANG = {
    "English": {
        "title": "AdikeRakshak AI",
        "subtitle": "Arecanut Disease Detection System",
        "language": "Select Language",
        "upload": "Upload Arecanut Leaf Image",
        "detect": "Detect Disease",
        "result": "Disease Detection Report",
        "disease": "Disease Name",
        "confidence": "AI Confidence",
        "medicine": "Medicine / Treatment",
        "area": "Estimated Affected Leaf Area",
        "dosage": "Medicine Dosage",
        "frequency": "Application Frequency",
        "symptoms": "Symptoms",
        "precautions": "Safety Precautions",
        "wait": "Analysing image...",
        "no_image": "Please upload an image first.",
        "not_verified": "Not verified yet",
        "low": "Low confidence. Please upload a clearer image.",
        "area_warning": "Experimental colour-based estimate. Not a validated disease measurement.",
        "footer": "AI results are preliminary. Confirm disease and treatment with an agricultural expert.",
        "healthy": "No disease treatment required for this healthy category.",
        "medicine_label": "Medicine / Product",
        "dosage_label": "Dosage",
        "frequency_note": "Follow the approved product label and agricultural expert advice.",
        "area_error": "Could not estimate the affected area from this image."
    },

    "ಕನ್ನಡ": {
        "title": "ಅಡಿಕೆ ರಕ್ಷಕ AI",
        "subtitle": "ಅಡಿಕೆ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
        "language": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "upload": "ಅಡಿಕೆ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "detect": "ರೋಗ ಪತ್ತೆ ಮಾಡಿ",
        "result": "ರೋಗ ಪತ್ತೆ ವರದಿ",
        "disease": "ರೋಗದ ಹೆಸರು",
        "confidence": "AI ವಿಶ್ವಾಸಮಟ್ಟ",
        "medicine": "ಔಷಧಿ / ಚಿಕಿತ್ಸೆ",
        "area": "ಅಂದಾಜು ಬಾಧಿತ ಎಲೆ ಪ್ರದೇಶ",
        "dosage": "ಔಷಧಿಯ ಪ್ರಮಾಣ",
        "frequency": "ಬಳಕೆಯ ಅವಧಿ",
        "symptoms": "ಲಕ್ಷಣಗಳು",
        "precautions": "ಸುರಕ್ಷತಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು",
        "wait": "ಚಿತ್ರವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...",
        "no_image": "ದಯವಿಟ್ಟು ಮೊದಲು ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "not_verified": "ಇನ್ನೂ ಪರಿಶೀಲಿಸಿಲ್ಲ",
        "low": "ವಿಶ್ವಾಸಮಟ್ಟ ಕಡಿಮೆ. ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "area_warning": "ಇದು ಬಣ್ಣ ಆಧಾರಿತ ಪ್ರಾಯೋಗಿಕ ಅಂದಾಜು ಮಾತ್ರ.",
        "footer": "AI ಫಲಿತಾಂಶಗಳು ಪ್ರಾಥಮಿಕವಾಗಿವೆ. ಕೃಷಿ ತಜ್ಞರಿಂದ ದೃಢೀಕರಿಸಿ.",
        "healthy": "ಆರೋಗ್ಯಕರ ವರ್ಗಕ್ಕೆ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.",
        "medicine_label": "ಔಷಧಿ / ಉತ್ಪನ್ನ",
        "dosage_label": "ಪ್ರಮಾಣ",
        "frequency_note": "ಅನುಮೋದಿತ ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಮತ್ತು ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆಯನ್ನು ಅನುಸರಿಸಿ.",
        "area_error": "ಈ ಚಿತ್ರದಿಂದ ಬಾಧಿತ ಪ್ರದೇಶವನ್ನು ಅಂದಾಜು ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ."
    },

    "हिन्दी": {
        "title": "अडिकेरक्षक AI",
        "subtitle": "सुपारी रोग पहचान प्रणाली",
        "language": "भाषा चुनें",
        "upload": "सुपारी के पत्ते की तस्वीर अपलोड करें",
        "detect": "रोग पहचानें",
        "result": "रोग पहचान रिपोर्ट",
        "disease": "रोग का नाम",
        "confidence": "AI विश्वास स्तर",
        "medicine": "दवा / उपचार",
        "area": "अनुमानित प्रभावित पत्ती का क्षेत्र",
        "dosage": "दवा की मात्रा",
        "frequency": "प्रयोग की आवृत्ति",
        "symptoms": "लक्षण",
        "precautions": "सुरक्षा सावधानियाँ",
        "wait": "तस्वीर का विश्लेषण हो रहा है...",
        "no_image": "कृपया पहले तस्वीर अपलोड करें।",
        "not_verified": "अभी सत्यापित नहीं",
        "low": "विश्वास स्तर कम है। कृपया स्पष्ट तस्वीर अपलोड करें।",
        "area_warning": "यह रंग के आधार पर प्रायोगिक अनुमान है।",
        "footer": "AI परिणाम प्रारंभिक हैं। कृषि विशेषज्ञ से पुष्टि करें।",
        "healthy": "स्वस्थ श्रेणी के लिए उपचार आवश्यक नहीं है।",
        "medicine_label": "दवा / उत्पाद",
        "dosage_label": "मात्रा",
        "frequency_note": "स्वीकृत उत्पाद के लेबल और कृषि विशेषज्ञ की सलाह का पालन करें।",
        "area_error": "इस तस्वीर से प्रभावित क्षेत्र का अनुमान नहीं लगाया जा सका।"
    },

    "తెలుగు": {
        "title": "అడికెరక్షక్ AI",
        "subtitle": "అరెకా వ్యాధి గుర్తింపు వ్యవస్థ",
        "language": "భాషను ఎంచుకోండి",
        "upload": "అరెకా ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "detect": "వ్యాధిని గుర్తించండి",
        "result": "వ్యాధి గుర్తింపు నివేదిక",
        "disease": "వ్యాధి పేరు",
        "confidence": "AI విశ్వాస స్థాయి",
        "medicine": "మందు / చికిత్స",
        "area": "ప్రభావిత ఆకు ప్రాంతం అంచనా",
        "dosage": "మందు మోతాదు",
        "frequency": "వాడే వ్యవధి",
        "symptoms": "లక్షణాలు",
        "precautions": "భద్రతా జాగ్రత్తలు",
        "wait": "చిత్రాన్ని విశ్లేషిస్తోంది...",
        "no_image": "ముందుగా చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "not_verified": "ఇంకా ధృవీకరించబడలేదు",
        "low": "విశ్వాస స్థాయి తక్కువ. స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",
        "area_warning": "ఇది రంగు ఆధారిత ప్రయోగాత్మక అంచనా మాత్రమే.",
        "footer": "AI ఫలితాలు ప్రాథమికమైనవి. వ్యవసాయ నిపుణుడితో నిర్ధారించండి.",
        "healthy": "ఆరోగ్యకరమైన వర్గానికి చికిత్స అవసరం లేదు.",
        "medicine_label": "మందు / ఉత్పత్తి",
        "dosage_label": "మోతాదు",
        "frequency_note": "ఆమోదిత ఉత్పత్తి లేబుల్ మరియు నిపుణుల సలహాను అనుసరించండి.",
        "area_error": "ఈ చిత్రం నుండి ప్రభావిత ప్రాంతాన్ని అంచనా వేయలేకపోయాము."
    },

    "தமிழ்": {
        "title": "அடைக்கரக்ஷக் AI",
        "subtitle": "பாக்கு நோய் கண்டறிதல் அமைப்பு",
        "language": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "upload": "பாக்கு இலைப் படத்தைப் பதிவேற்றவும்",
        "detect": "நோயைக் கண்டறியவும்",
        "result": "நோய் கண்டறிதல் அறிக்கை",
        "disease": "நோயின் பெயர்",
        "confidence": "AI நம்பகத்தன்மை",
        "medicine": "மருந்து / சிகிச்சை",
        "area": "பாதிக்கப்பட்ட இலையின் மதிப்பிடப்பட்ட பகுதி",
        "dosage": "மருந்தின் அளவு",
        "frequency": "பயன்பாட்டு இடைவெளி",
        "symptoms": "அறிகுறிகள்",
        "precautions": "பாதுகாப்பு முன்னெச்சரிக்கைகள்",
        "wait": "படம் பகுப்பாய்வு செய்யப்படுகிறது...",
        "no_image": "முதலில் ஒரு படத்தைப் பதிவேற்றவும்.",
        "not_verified": "இன்னும் சரிபார்க்கப்படவில்லை",
        "low": "நம்பகத்தன்மை குறைவு. தெளிவான படத்தைப் பதிவேற்றவும்.",
        "area_warning": "இது நிறத்தின் அடிப்படையிலான சோதனை மதிப்பீடு மட்டுமே.",
        "footer": "AI முடிவுகள் ஆரம்பநிலை மட்டுமே. வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும்.",
        "healthy": "ஆரோக்கியமான வகைக்கு சிகிச்சை தேவையில்லை.",
        "medicine_label": "மருந்து / தயாரிப்பு",
        "dosage_label": "அளவு",
        "frequency_note": "அங்கீகரிக்கப்பட்ட தயாரிப்பு லேபிள் மற்றும் நிபுணர் ஆலோசனையைப் பின்பற்றவும்.",
        "area_error": "இந்தப் படத்திலிருந்து பாதிக்கப்பட்ட பகுதியை மதிப்பிட முடியவில்லை."
    },

    "മലയാളം": {
        "title": "അടിക്കെരക്ഷക് AI",
        "subtitle": "അടയ്ക്ക രോഗനിർണയ സംവിധാനം",
        "language": "ഭാഷ തിരഞ്ഞെടുക്കുക",
        "upload": "അടയ്ക്ക ഇലയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "detect": "രോഗം കണ്ടെത്തുക",
        "result": "രോഗനിർണയ റിപ്പോർട്ട്",
        "disease": "രോഗത്തിന്റെ പേര്",
        "confidence": "AI വിശ്വാസനില",
        "medicine": "മരുന്ന് / ചികിത്സ",
        "area": "ബാധിച്ച ഇലയുടെ ഏകദേശ വിസ്തീർണ്ണം",
        "dosage": "മരുന്നിന്റെ അളവ്",
        "frequency": "പ്രയോഗത്തിന്റെ ഇടവേള",
        "symptoms": "ലക്ഷണങ്ങൾ",
        "precautions": "സുരക്ഷാ മുൻകരുതലുകൾ",
        "wait": "ചിത്രം പരിശോധിക്കുന്നു...",
        "no_image": "ആദ്യം ഒരു ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "not_verified": "ഇതുവരെ സ്ഥിരീകരിച്ചിട്ടില്ല",
        "low": "വിശ്വാസനില കുറവാണ്. വ്യക്തമായ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "area_warning": "ഇത് നിറത്തെ അടിസ്ഥാനമാക്കിയുള്ള പരീക്ഷണാത്മക കണക്കുകൂട്ടലാണ്.",
        "footer": "AI ഫലങ്ങൾ പ്രാഥമികമാണ്. കാർഷിക വിദഗ്ധനോട് സ്ഥിരീകരിക്കുക.",
        "healthy": "ആരോഗ്യമുള്ള വിഭാഗത്തിന് ചികിത്സ ആവശ്യമില്ല.",
        "medicine_label": "മരുന്ന് / ഉൽപ്പന്നം",
        "dosage_label": "അളവ്",
        "frequency_note": "അംഗീകൃത ഉൽപ്പന്ന ലേബലും വിദഗ്ധ ഉപദേശവും പാലിക്കുക.",
        "area_error": "ഈ ചിത്രത്തിൽ നിന്ന് ബാധിച്ച ഭാഗം കണക്കാക്കാൻ കഴിഞ്ഞില്ല."
    },

    "ਪੰਜਾਬੀ": {
        "title": "ਅਡਿਕੇਰਕਸ਼ਕ AI",
        "subtitle": "ਸੁਪਾਰੀ ਰੋਗ ਪਛਾਣ ਪ੍ਰਣਾਲੀ",
        "language": "ਭਾਸ਼ਾ ਚੁਣੋ",
        "upload": "ਸੁਪਾਰੀ ਦੇ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ",
        "detect": "ਰੋਗ ਦੀ ਪਛਾਣ ਕਰੋ",
        "result": "ਰੋਗ ਪਛਾਣ ਰਿਪੋਰਟ",
        "disease": "ਰੋਗ ਦਾ ਨਾਮ",
        "confidence": "AI ਭਰੋਸੇ ਦਾ ਪੱਧਰ",
        "medicine": "ਦਵਾਈ / ਇਲਾਜ",
        "area": "ਪ੍ਰਭਾਵਿਤ ਪੱਤੇ ਦਾ ਅੰਦਾਜ਼ਨ ਖੇਤਰ",
        "dosage": "ਦਵਾਈ ਦੀ ਮਾਤਰਾ",
        "frequency": "ਵਰਤੋਂ ਦੀ ਆਵਰਤੀ",
        "symptoms": "ਲੱਛਣ",
        "precautions": "ਸੁਰੱਖਿਆ ਸਾਵਧਾਨੀਆਂ",
        "wait": "ਤਸਵੀਰ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
        "no_image": "ਕਿਰਪਾ ਕਰਕੇ ਪਹਿਲਾਂ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।",
        "not_verified": "ਹਾਲੇ ਤਸਦੀਕ ਨਹੀਂ ਹੋਈ",
        "low": "ਭਰੋਸੇ ਦਾ ਪੱਧਰ ਘੱਟ ਹੈ। ਸਾਫ਼ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ।",
        "area_warning": "ਇਹ ਰੰਗ ਦੇ ਆਧਾਰ 'ਤੇ ਪ੍ਰਯੋਗਾਤਮਕ ਅੰਦਾਜ਼ਾ ਹੈ।",
        "footer": "AI ਨਤੀਜੇ ਸ਼ੁਰੂਆਤੀ ਹਨ। ਖੇਤੀਬਾੜੀ ਮਾਹਿਰ ਤੋਂ ਪੁਸ਼ਟੀ ਕਰੋ।",
        "healthy": "ਸਿਹਤਮੰਦ ਸ਼੍ਰੇਣੀ ਲਈ ਇਲਾਜ ਦੀ ਲੋੜ ਨਹੀਂ ਹੈ।",
        "medicine_label": "ਦਵਾਈ / ਉਤਪਾਦ",
        "dosage_label": "ਮਾਤਰਾ",
        "frequency_note": "ਮਨਜ਼ੂਰਸ਼ੁਦਾ ਉਤਪਾਦ ਦੇ ਲੇਬਲ ਅਤੇ ਮਾਹਿਰ ਦੀ ਸਲਾਹ ਦੀ ਪਾਲਣਾ ਕਰੋ।",
        "area_error": "ਇਸ ਤਸਵੀਰ ਤੋਂ ਪ੍ਰਭਾਵਿਤ ਖੇਤਰ ਦਾ ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਗਾਇਆ ਜਾ ਸਕਿਆ।"
    },

    "हरियाणवी": {
        "title": "अडिकेरक्षक AI",
        "subtitle": "सुपारी की बीमारी पहचान प्रणाली",
        "language": "भाषा चुनो",
        "upload": "सुपारी के पत्ते की फोटो अपलोड करो",
        "detect": "बीमारी पहचानो",
        "result": "बीमारी की रिपोर्ट",
        "disease": "बीमारी का नाम",
        "confidence": "AI भरोसे का स्तर",
        "medicine": "दवाई / इलाज",
        "area": "प्रभावित पत्ते का अनुमानित हिस्सा",
        "dosage": "दवाई की मात्रा",
        "frequency": "दवाई डालने का समय",
        "symptoms": "लक्षण",
        "precautions": "सुरक्षा की सावधानियां",
        "wait": "फोटो की जांच हो री है...",
        "no_image": "पहले फोटो अपलोड करो।",
        "not_verified": "अभी जांच नहीं हुई",
        "low": "भरोसे का स्तर कम है। साफ फोटो अपलोड करो।",
        "area_warning": "यो रंग के आधार पर लगाया गया प्रयोगात्मक अनुमान है।",
        "footer": "AI का नतीजा शुरुआती है। खेती के माहिर से पुष्टि कराओ।",
        "healthy": "स्वस्थ श्रेणी में इलाज की जरूरत नहीं है।",
        "medicine_label": "दवाई / उत्पाद",
        "dosage_label": "मात्रा",
        "frequency_note": "मंजूर उत्पाद के लेबल और खेती के माहिर की सलाह मानो।",
        "area_error": "इस फोटो से प्रभावित हिस्सा नहीं नापा जा सका।"
    }
}

# ---------------- DISEASE NAMES ----------------

DISEASE_NAMES = {
    "mahali_koleroga": {
        "English": "Mahali (Koleroga)",
        "ಕನ್ನಡ": "ಮಹಾಳಿ (ಕೊಳೆರೋಗ)",
        "हिन्दी": "महाली (फल सड़न रोग)",
        "తెలుగు": "మహాలి (కాయ కుళ్ళు తెగులు)",
        "தமிழ்": "மகாலி (காய் அழுகல் நோய்)",
        "മലയാളം": "മഹാളി (കായ്ചീയൽ രോഗം)",
        "ਪੰਜਾਬੀ": "ਮਹਾਲੀ (ਫਲ ਸੜਨ ਦੀ ਬਿਮਾਰੀ)",
        "हरियाणवी": "महाली (फल सड़न की बीमारी)"
    },
    "yellow_leaf_disease": {
        "English": "Yellow Leaf Disease",
        "ಕನ್ನಡ": "ಹಳದಿ ಎಲೆ ರೋಗ",
        "हिन्दी": "पीली पत्ती रोग",
        "తెలుగు": "పసుపు ఆకు తెగులు",
        "தமிழ்": "மஞ்சள் இலை நோய்",
        "മലയാളം": "മഞ്ഞളിപ്പ് രോഗം",
        "ਪੰਜਾਬੀ": "ਪੀਲੇ ਪੱਤੇ ਦੀ ਬਿਮਾਰੀ",
        "हरियाणवी": "पीले पत्ते की बीमारी"
    },
    "stem_bleeding": {
        "English": "Stem Bleeding",
        "ಕನ್ನಡ": "ಕಾಂಡದಿಂದ ರಸ ಸೋರುವಿಕೆ",
        "हिन्दी": "तने से रस निकलना",
        "తెలుగు": "కాండం నుండి రసం కారడం",
        "தமிழ்": "தண்டு கசிவு நோய்",
        "മലയാളം": "തണ്ടിൽ നിന്ന് നീരൊലിപ്പ്",
        "ਪੰਜਾਬੀ": "ਤਣੇ ਵਿੱਚੋਂ ਰਸ ਨਿਕਲਣਾ",
        "हरियाणवी": "तने से रस निकलना"
    },
    "bud_borer": {
        "English": "Bud Borer",
        "ಕನ್ನಡ": "ಮೊಗ್ಗು ಕೊರಕ",
        "हिन्दी": "कली छेदक कीट",
        "తెలుగు": "మొగ్గ తొలిచే పురుగు",
        "தமிழ்": "மொட்டு துளைப்பான்",
        "മലയാളം": "മൊട്ട് തുരപ്പൻ",
        "ਪੰਜਾਬੀ": "ਕਲੀ ਛੇਦਕ ਕੀੜਾ",
        "हरियाणवी": "कली में छेद करने वाला कीड़ा"
    },
    "healthy_leaf": {
        "English": "Healthy Leaf",
        "ಕನ್ನಡ": "ಆರೋಗ್ಯಕರ ಎಲೆ",
        "हिन्दी": "स्वस्थ पत्ती",
        "తెలుగు": "ఆరోగ్యకరమైన ఆకు",
        "தமிழ்": "ஆரோக்கியமான இலை",
        "മലയാളം": "ആരോഗ്യമുള്ള ഇല",
        "ਪੰਜਾਬੀ": "ਸਿਹਤਮੰਦ ਪੱਤਾ",
        "हरियाणवी": "स्वस्थ पत्ता"
    },
    "healthy_nut": {
        "English": "Healthy Nut",
        "ಕನ್ನಡ": "ಆರೋಗ್ಯಕರ ಅಡಿಕೆ",
        "हिन्दी": "स्वस्थ सुपारी",
        "తెలుగు": "ఆరోగ్యకరమైన వక్క",
        "தமிழ்": "ஆரோக்கியமான பாக்கு",
        "മലയാളം": "ആരോഗ്യമുള്ള അടയ്ക്ക",
        "ਪੰਜਾਬੀ": "ਸਿਹਤਮੰਦ ਸੁਪਾਰੀ",
        "हरियाणवी": "स्वस्थ सुपारी"
    },
    "healthy_trunk": {
        "English": "Healthy Trunk",
        "ಕನ್ನಡ": "ಆರೋಗ್ಯಕರ ಕಾಂಡ",
        "हिन्दी": "स्वस्थ तना",
        "తెలుగు": "ఆరోగ్యకరమైన కాండం",
        "தமிழ்": "ஆரோக்கியமான தண்டு",
        "മലയാളം": "ആരോഗ്യമുള്ള തടി",
        "ਪੰਜਾਬੀ": "ਸਿਹਤਮੰਦ ਤਣਾ",
        "हरियाणवी": "स्वस्थ तना"
    },
    "healthy_foot": {
        "English": "Healthy Palm Base",
        "ಕನ್ನಡ": "ಆರೋಗ್ಯಕರ ಬುಡ",
        "हिन्दी": "स्वस्थ पेड़ का आधार",
        "తెలుగు": "ఆరోగ్యకరమైన చెట్టు అడుగు భాగం",
        "தமிழ்": "ஆரோக்கியமான அடிப்பகுதி",
        "മലയാളം": "ആരോഗ്യമുള്ള അടിഭാഗം",
        "ਪੰਜਾਬੀ": "ਸਿਹਤਮੰਦ ਰੁੱਖ ਦਾ ਹੇਠਲਾ ਹਿੱਸਾ",
        "हरियाणवी": "स्वस्थ पेड़ का निचला हिस्सा"
    },
    "stem_cracking": {
        "English": "Stem Cracking",
        "ಕನ್ನಡ": "ಕಾಂಡ ಬಿರುಕು",
        "हिन्दी": "तने में दरार",
        "తెలుగు": "కాండం పగుళ్లు",
        "தமிழ்": "தண்டு வெடிப்பு",
        "മലയാളം": "തണ്ടിലെ വിള്ളൽ",
        "ਪੰਜਾਬੀ": "ਤਣੇ ਵਿੱਚ ਤਰੇੜ",
        "हरियाणवी": "तने में दरार"
    }
}

# ---------------- HELPER FUNCTIONS ----------------

def normalise(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")

TREATMENTS = {
    normalise(key): value
    for key, value in TREATMENT_DATA.items()
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        try:
            return [
                str(data[k])
                for k in sorted(data, key=lambda x: int(x))
            ]
        except (ValueError, TypeError):
            return [str(v) for v in data.values()]

    raise ValueError("Invalid classes.json format")

# ---------------- LOAD MODEL ----------------

try:
    if not MODEL_PATH.exists():
        st.error("arecanut_model.keras is missing.")
        st.stop()

    if not CLASSES_PATH.exists():
        st.error("classes.json is missing.")
        st.stop()

    model = load_model()
    classes = load_classes()

    if model.output_shape[-1] != len(classes):
        st.error("Model and class count do not match.")
        st.stop()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- AFFECTED AREA ESTIMATE ----------------

def estimate_affected_area(pil_image):
    """
    Experimental colour-based estimate.
    This is NOT a trained disease segmentation model.
    """

    try:
        import cv2

        image = np.array(pil_image.convert("RGB"))
        image = cv2.resize(image, (400, 400))

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = np.zeros(image.shape[:2], np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        height, width = mask.shape
        rect = (5, 5, width - 10, height - 10)

        cv2.grabCut(
            bgr, mask, rect,
            bg_model, fg_model,
            5, cv2.GC_INIT_WITH_RECT
        )

        leaf_mask = np.where(
            (mask == 1) | (mask == 3),
            1, 0
        ).astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        affected_mask = (
            (hue >= 5) &
            (hue <= 35) &
            (saturation > 45) &
            (brightness > 35)
        )

        total_pixels = np.sum(leaf_mask)

        if total_pixels == 0:
            return None

        affected_pixels = np.sum(
            affected_mask & (leaf_mask == 1)
        )

        return round(
            float(affected_pixels / total_pixels * 100),
            2
        )

    except Exception:
        return None

# ---------------- INTERFACE STYLE ----------------

st.markdown("""
<style>
.stApp {
    background-color: #071810;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background-color: #37b879;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    min-height: 45px;
}
[data-testid="stFileUploader"] {
    background-color: #10271c;
    border-radius: 12px;
    padding: 12px;
}
.result-card {
    background-color: #10271c;
    border: 1px solid #39875d;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MAIN APP ----------------

language = st.selectbox(
    "🌐 Select Language / ಭಾಷೆ / भाषा",
    list(LANG.keys())
)

t = LANG[language]

st.title("🌴 " + t["title"])
st.write(t["subtitle"])
st.divider()

st.subheader("📷 " + t["upload"])

uploaded = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption=t["upload"],
        use_container_width=True
    )

    if st.button("🔍 " + t["detect"], use_container_width=True):

        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = int(input_shape[1] or 224)
        width = int(input_shape[2] or 224)

        resized = image.resize((width, height))

        array = np.asarray(
            resized,
            dtype=np.float32
        )

        array = np.expand_dims(array, axis=0)

        with st.spinner(t["wait"]):
            prediction = model.predict(array, verbose=0)[0]

            index = int(np.argmax(prediction))
            disease = classes[index]
            confidence = float(prediction[index]) * 100

            affected_area = estimate_affected_area(image)

        st.session_state["result"] = {
            "disease": disease,
            "confidence": confidence,
            "image": image.copy(),
            "affected_area": affected_area
        }

# ---------------- COMPLETE ML DISEASE REPORT ----------------

if "result" in st.session_state:

    result = st.session_state["result"]

    disease = result["disease"]
    confidence = result["confidence"]
    disease_key = normalise(disease)

    translated_disease = DISEASE_NAMES.get(
        disease_key, {}
    ).get(language, disease)

    treatment = TREATMENTS.get(disease_key)

    st.divider()
    st.header("📋 " + t["result"])

    st.image(
        result["image"],
        caption=translated_disease,
        use_container_width=True
    )

    # 1. DISEASE NAME

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("1. 🦠 " + t["disease"])
    st.success(translated_disease)

    st.metric(
        t["confidence"],
        f"{confidence:.2f}%"
    )

    if confidence < 60:
        st.warning(t["low"])

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. MEDICINE / TREATMENT

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("2. 💊 " + t["medicine"])

    if disease_key.startswith("healthy_"):
        st.success(t["healthy"])

    elif treatment:
        st.write(
            treatment.get("treatment", t["not_verified"])
        )

        st.write(
            "**" + t["symptoms"] + ":**",
            treatment.get("symptoms", t["not_verified"])
        )

        st.write(
            "**" + t["medicine_label"] + ":**",
            treatment.get("medicine", t["not_verified"])
        )

        if treatment.get("source"):
            st.caption("Reference: " + treatment["source"])
        if treatment.get("source_url"):
            st.markdown("[Open agricultural reference](" + treatment["source_url"] + ")")

    else:
        st.warning(t["not_verified"])

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. AFFECTED AREA

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("3. 📊 " + t["area"])

    if result["affected_area"] is not None:
        percentage = result["affected_area"]

        st.metric(
            t["area"],
            f"{percentage:.2f}%"
        )

        st.progress(
            min(100, max(0, int(round(percentage))))
        )

    else:
        st.warning(t["area_error"])

    st.caption(t["area_warning"])
    st.info(
        "This is a colour-based visual estimate, not an exact disease-area measurement. "
        "It may count natural brown/yellow colour, shadows or background errors. "
        "Validated automatic measurement requires a segmentation model trained with "
        "pixel-level annotated arecanut images."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. DOSAGE AND FREQUENCY

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("4. 🧪 " + t["dosage"] + " / Application Schedule")

    if disease_key.startswith("healthy_"):
        st.success(t["healthy"])

    elif treatment:
        st.write(
            "**" + t["dosage_label"] + ":**",
            treatment.get("dosage", t["not_verified"])
        )

        st.write(
            "**" + t["frequency"] + ":**",
            treatment.get("frequency", t["not_verified"])
        )

        # Calculate quantity only when a verified label rate is supplied.
        rate = treatment.get("label_rate_ml_per_litre")
        if rate is not None:
            st.markdown("**Label-based tank quantity calculator**")
            tank_litres = st.number_input(
                "Spray tank capacity (litres)",
                min_value=1.0,
                max_value=500.0,
                value=15.0,
                step=1.0,
                key="tank_capacity"
            )
            st.info(
                f"Calculated quantity: {rate * tank_litres:.2f} ml "
                f"for {tank_litres:g} L water. Use only if this exact "
                "product formulation and crop use match its current approved label."
            )
        else:
            st.caption(
                "No ml/L product-label rate is stored for this recommendation. "
                "Do not convert a mixture percentage into a product dose."
            )

        st.write(
            "**" + t["precautions"] + ":**",
            treatment.get("precautions", t["frequency_note"])
        )

    else:
        st.warning(t["not_verified"])

    st.markdown("</div>", unsafe_allow_html=True)

    # 5. WHEN, WHY AND SEASONAL RISK
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("5. 🗓️ When, Why and Seasonal Risk")
    if disease_key.startswith("healthy_"):
        st.write("**When:** No disease symptoms are indicated in this image. Continue monitoring throughout the year.")
        st.write("**Why:** A healthy classification does not rule out early or hidden problems elsewhere on the palm.")
        st.write("**Seasonal risk:** Not applicable to this healthy prediction. Continue extra checks during prolonged wet weather.")
    elif treatment:
        st.write("**When it occurs:**", treatment.get("when", "Timing is not established in the stored advisory; inspect regularly."))
        st.write("**Why it occurs:**", treatment.get("why", "The cause needs confirmation by a local agriculture expert."))
        seasonal = {
            "mahali_koleroga": "Highest concern during the southwest monsoon, commonly June–September in Karnataka, and other prolonged wet periods.",
            "bud_rot": "Risk is higher during rainy, humid periods when the crown remains wet. Exact local timing varies.",
            "yellow_leaf_disease": "No single high-risk season is established here; symptoms may progress gradually through the year.",
            "stem_bleeding": "No reliable seasonal peak is established for this image class.",
            "bud_borer": "A specific seasonal window cannot be stated without identifying the insect species and local pest cycle.",
            "stem_cracking": "No disease season applies; cracks may follow injury or other stresses at any time."
        }
        st.write("**Season when risk is higher:**", seasonal.get(disease_key, "Seasonal pattern not verified for this class."))
    else:
        st.warning("Season and cause information are not available for this class.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. AFFECTED PLANT PARTS
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("6. 🌿 Symptoms and Affected Plant Parts")
    affected_parts = {
        "mahali_koleroga": "Immature nuts and bunches; fruit surface and developing nuts.",
        "yellow_leaf_disease": "Leaflets and fronds; progressive yellowing may affect the canopy and whole palm.",
        "stem_bleeding": "Trunk/stem tissue.",
        "bud_rot": "Crown, spindle leaf and growing bud.",
        "bud_borer": "Bud/crown area; confirm the insect and feeding damage in the field.",
        "stem_cracking": "Trunk/stem surface and underlying tissue.",
        "healthy_leaf": "Leaf image classified as healthy.",
        "healthy_nut": "Nut image classified as healthy.",
        "healthy_trunk": "Trunk image classified as healthy.",
        "healthy_foot": "Palm base/foot image classified as healthy."
    }
    st.write("**Affected part(s):**", affected_parts.get(disease_key, "Not specified; inspect the whole palm."))
    if treatment:
        st.write("**Symptoms:**", treatment.get("symptoms", "Not available in the advisory data."))
    st.caption("The image classifier predicts an image category; it does not independently confirm the causal organism or field diagnosis.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 7. EXPECTED IMPROVEMENT / RECOVERY
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("7. ⏳ Expected Improvement / Recovery")
    if treatment:
        st.write(treatment.get("recovery_time", "A reliable recovery estimate is not available. Field assessment is needed."))
    else:
        st.write("A recovery estimate is not available for this class.")
    st.caption("This is guidance, not a guaranteed cure date. Damaged or rotted plant tissue may not recover.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 8. PREVENTIVE FERTILIZER AND NUTRITION
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("8. 🌱 Fertilizer and Nutrition Before Disease Occurrence")
    st.write("**General bearing-palm reference:** ICAR-CPCRI's general recommendation for a bearing arecanut palm is 100 g N, 40 g P₂O₅ and 140 g K₂O per palm per year, along with 12 kg compost/FYM and 12 kg green leaves per palm per year.")
    st.write("**Application note:** Adjust the recommendation to palm age, soil-test results, garden condition and local ICAR-CPCRI/KVK/UAS advice. Do not apply the full bearing-palm dose automatically to young palms.")
    if disease_key == "yellow_leaf_disease":
        st.info("For gardens affected by Yellow Leaf Disease, the stored ICAR-CPCRI advisory also describes an integrated nutrition programme: 220 g urea, 200 g superphosphate and 230 g muriate of potash per palm annually in two splits, plus 1 kg superphosphate per palm in affected gardens and 12 kg each of compost and green leaves per palm per year. Confirm current local advice and soil needs before applying.")
    st.warning("Fertilizer supports balanced crop nutrition and prevention; it is not a cure for an established infection. Do not use nutrition advice as a substitute for disease diagnosis.")
    st.caption("Reference: ICAR-CPCRI arecanut crop advice. Verify the recommendation for the palm's age and local soil before use.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 9. PREVENTION AND REGULAR CARE
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("9. 🛡️ Prevention and Regular Care")
    prevention = {
        "mahali_koleroga": ["Improve drainage and avoid prolonged waterlogging.", "Cover arecanut bunches before the monsoon where recommended.", "Remove and safely dispose of fallen infected nuts.", "Follow the locally verified pre-monsoon protection schedule."],
        "yellow_leaf_disease": ["Maintain balanced nutrition based on soil testing.", "Ensure good drainage and avoid water stress.", "Monitor nearby palms and seek field confirmation because nutrient disorders can look similar."],
        "stem_bleeding": ["Avoid trunk injuries and maintain garden sanitation.", "Check drainage and have trunk oozing examined before any chemical treatment."],
        "bud_rot": ["Prevent persistent water accumulation in the crown.", "Inspect the spindle and crown regularly during wet weather.", "Seek prompt expert help if the growing bud is affected."],
        "bud_borer": ["Inspect the crown and bud regularly for fresh feeding signs.", "Get the insect identified before selecting any pesticide.", "Use only a current approved crop-specific label recommendation."],
        "stem_cracking": ["Avoid mechanical injury to the trunk.", "Keep the palm base and garden clean.", "Have deep or spreading cracks inspected to identify the cause."],
        "healthy_leaf": ["Monitor leaves regularly.", "Maintain suitable irrigation, drainage and balanced nutrition.", "Remove dead or damaged plant material."],
        "healthy_nut": ["Inspect bunches during wet weather.", "Maintain field hygiene and remove fallen diseased nuts.", "Harvest and store nuts under clean, suitable conditions."],
        "healthy_trunk": ["Inspect the trunk regularly.", "Avoid mechanical injury.", "Prevent water stagnation around the palm."],
        "healthy_foot": ["Maintain drainage around the palm base.", "Keep the base clean and monitor for unusual rot or fungal growth."]
    }
    for item in prevention.get(disease_key, ["Monitor the palm regularly.", "Maintain drainage, field sanitation and balanced nutrition.", "Consult a local horticulture officer if symptoms progress."]):
        st.markdown("- " + item)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption(t["footer"])

else:
    st.info(t["no_image"])
