import random
import json
import subprocess
import joblib
import pyaudio
import os
import tempfile
import time
from datetime import datetime
from vosk import Model, KaldiRecognizer

# =====================================================
# CONFIG (✅ your desired changes applied)
# =====================================================

# ✅ Wake words (you can add more)
WAKE_WORDS = ["विजय", "सुनो", "हेलो", "hello", "wake", "बॉट"]

# ✅ Sleep after 15 seconds of silence (as you asked)
SLEEP_TIMEOUT = 15

# ✅ Optional: words to fully exit the program (hard off)
EXIT_WORDS = {"exit", "quit", "stop", "बंद", "ऑफ", "बाय", "shutdown"}

VOSK_MODEL_PATH = "vosk-model-small-hi-0.22"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# TTS USING PIPER (kept as your code)
# =====================================================
PIPER_BIN = os.path.join(BASE_DIR, "piper", "piper.exe")
PIPER_MODEL = os.path.join(BASE_DIR, "piper", "voices", "hi_IN-rohan-medium.onnx")

def speak(text: str):
    print("Bot:", text)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", wav_path],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    subprocess.run(
        ["powershell", "-c",
         f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync();"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    os.remove(wav_path)

# =====================================================
# LOAD VOSK + MIC
# =====================================================
model_vosk = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(model_vosk, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=4096
)
stream.start_stream()

def listen():
    """Return final recognized text when a phrase completes, else ''."""
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        return (result.get("text") or "").strip()
    return ""

# =====================================================
# LOAD ML MODEL
# =====================================================
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_intent(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# =====================================================
# RESPONSES
# =====================================================
RESPONSES = {
    # "greeting": ["नमस्ते! मैं सुन रहा हूँ।"],
    # "goodbye": ["ठीक है, फिर मिलेंगे।"],
    # "thank_you": ["आपका स्वागत है।"],
    # "emergency": ["आपातकाल है तो तुरंत नज़दीकी मदद लें या हेल्पलाइन पर कॉल करें।"],

    # "fever_info": ["बुखार में आराम करें और पानी तथा ओआरएस लेते रहें। अगर ज्यादा हो तो डॉक्टर से मिलें।"],
    # "cold_cough_info": ["सर्दी-खांसी में गरम पानी, भाप और आराम मदद करता है। ज्यादा हो तो डॉक्टर से मिलें।"],
    # "headache_info": ["सिर दर्द में आराम करें, पानी पिएँ। बहुत तेज हो तो डॉक्टर से मिलें।"],
    # "acidity_info": ["एसिडिटी में हल्का भोजन करें। ज्यादा हो तो डॉक्टर से सलाह लें।"],
    # "motion_info": ["दस्त में ORS/पानी लें। ज्यादा हो तो डॉक्टर से मिलें।"],
    # "weakness_symptom": ["कमजोरी में आराम, पानी और पौष्टिक भोजन लें।"],

    # "crop_info": ["आप फसल से जुड़ी जानकारी पूछ रहे हैं।"],
    # "fertilizer_info": ["आप खाद/मात्रा/संतुलन से जुड़ी बात पूछ रहे हैं।"],
    # "pest_problem": ["कीड़े/नुकसान/रोक से जुड़ी समस्या लग रही है।"],

    # "electricity_info": ["बिजली/लाइट समस्या लग रही है।"],
    # "water_problem": ["पानी सप्लाई की समस्या लग रही है।"],
    # "document_help": ["दस्तावेज/कार्ड से जुड़ी सहायता चाहिए।"],

    # "unknown": ["माफ़ कीजिए, मैं समझ नहीं पाया।"]
    "greeting": ["नमस्ते"],
    "goodbye": ["अलविदा, फिर मिलेंगे"],
    "thank_you": ["आपकी मदद करके खुशी हुई"],
    "emergency": ["तुरंत मदद के लिए 108 या 112 पर संपर्क करें"],

    "fever_info": ["बुखार में आराम करें और पानी तथा ओआरएस लेते रहें। अगर बुखार 2 दिन से ज्यादा रहे या बहुत तेज हो तो डॉक्टर से संपर्क करें।"],
    "cold_cough_info": ["सर्दी-खांसी में गरम पानी, भाप और आराम मदद कर सकता है। सांस फूल रही हो तो डॉक्टर से मिलें।"],
    "headache_info": ["सिर दर्द में पानी पिएं और थोड़ा आराम करें तथा स्क्रीन समय कम करें। ज्यादा सिर दर्द हो तो डॉक्टर से संपर्क करें।"],
    "acidity_info": ["एसिडिटी में तला-भुना और मसालेदार भोजन कम करें, समय पर खाना खाएं और पानी पिएं।"],
    "motion_info": ["दस्त में ओआरएस और पानी बहुत जरूरी है। हल्का भोजन करें।"],
    "weakness_symptom": ["अगर थकान लंबे समय से है तो शुगर और बीपी की जांच कराएं।"],

    "crop_info": ["बरसात के मौसम में धान, मक्का और सोयाबीन उगाए जाते हैं। सर्दी के मौसम में गेहूं, चना और सरसों। गर्मी के मौसम में खरबूजा, तरबूज और ककड़ी।"],
    "fertilizer_info": ["फसल के अनुसार ही खाद डालनी चाहिए। अधिक जानकारी के लिए कृषि सेवा केंद्र से संपर्क करें।"],
    "pest_problem": ["दवा डालने से पहले कीट की पहचान जरूरी है, इसलिए कृषि विभाग से संपर्क करें।"],

    "electricity_info": ["बिजली विभाग के लिए 1912 या 1910 पर संपर्क करें या ग्राम पंचायत कार्यालय से संपर्क करें।"],
    "water_problem": ["नगर पालिका हेल्पलाइन 1916 पर संपर्क करें या ग्राम पंचायत अथवा सरपंच से संपर्क करें।"],
    "document_help": ["आप नजदीकी जन सेवा केंद्र या तहसील कार्यालय में संपर्क करें।"],

    "unknown": ["मुझे समझ नहीं आया।"]
}

# =====================================================
# TIME / DATE / DAY
# =====================================================
def get_time():
    return datetime.now().strftime("समय: %I:%M %p")

def get_date():
    return datetime.now().strftime(" तारीख: %d-%m-%Y")

def get_day():
    return datetime.now().strftime(" दिन: %A")

# =====================================================
# HELPERS: Wake / Sleep logic
# =====================================================
def contains_wake_word(text: str) -> bool:
    # substring match (good for VOSK: "विजय सुनो", "hello bot", etc.)
    return any(w in text for w in WAKE_WORDS)

def pick_response(intent: str) -> str:
    resp = RESPONSES.get(intent, RESPONSES["unknown"])
    return random.choice(resp) if isinstance(resp, list) else str(resp)

# =====================================================
# MAIN LOOP
# =====================================================
print("\nAssistant Sleep Mode में है...")
print(f"Wake words: {', '.join(WAKE_WORDS)}")
print("15 सेकंड चुप्पी पर Sleep Mode में चला जाएगा.\n")

conversation_active = False
last_interaction_time = time.time()

while True:
    text = listen()

    # ✅ Silence handling: sleep after 15s of no recognized phrase
    if text == "":
        if conversation_active and (time.time() - last_interaction_time > SLEEP_TIMEOUT):
            conversation_active = False
            print("\n😴 Sleep Mode...\n")
            speak("ठीक है, मैं सो रहा हूँ।")
        continue

    print("Detected:", text)

    # ✅ Always allow hard-exit commands (optional)
    if text.lower() in EXIT_WORDS:
        speak("ठीक है, बंद कर रहा हूँ।")
        raise SystemExit

    # ✅ Wake-up handling (only wake words work in sleep mode)
    if not conversation_active:
        if contains_wake_word(text):
            conversation_active = True
            last_interaction_time = time.time()
            speak("नमस्ते! बोलिए।")
        # ignore everything else while sleeping
        continue

    # ✅ If awake: update timer
    last_interaction_time = time.time()

    # ✅ Predict intent
    intent = predict_intent(text)

    # ✅ Goodbye intent should put bot to sleep (as you asked)
    if intent == "goodbye":
        reply = pick_response("goodbye")
        speak(reply)
        conversation_active = False
        print("\n Sleep Mode (goodbye)...\n")
        speak("ठीक है, मैं सो रहा हूँ।")
        continue

    # ✅ Special intents
    if intent == "time_query":
        reply = get_time()
    elif intent == "date_query":
        reply = get_date()
    elif intent == "day_query":
        reply = get_day()
    else:
        reply = pick_response(intent)

    speak(reply)