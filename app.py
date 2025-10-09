# app.py — komplett ersättningsfil

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------ Flask + CORS ------------------------
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": [
        "https://swedoc.github.io",
        "https://7p84qiqowa0cp5.proxy.runpod.net"
    ]}},
    supports_credentials=True,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

# ------------------------ Konfig via miljövariabler ------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")          # small | medium | large-v3
LOG_TRANSCRIPT = os.getenv("LOG_TRANSCRIPT", "0") == "1"      # 1 för att logga transkription i terminalen
LANGUAGE = os.getenv("LANGUAGE", "sv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = (DEVICE == "cuda")

# ------------------------ Ladda Whisper globalt ------------------------
print(f"[BOOT] Laddar Whisper: {WHISPER_MODEL} på {DEVICE} (fp16={FP16})")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

def _transcribe_file(file_path: str, language: str = LANGUAGE):
    t0 = time.time()
    result = whisper_model.transcribe(
        file_path,
        language=language,
        fp16=FP16,
        condition_on_previous_text=False,
        temperature=0.0
    )
    t1 = time.time()

    # Hämta uppskattad ljudlängd från sista segmentet
    audio_sec = None
    try:
        segs = result.get("segments", [])
        if segs:
            audio_sec = float(segs[-1]["end"])
    except Exception:
        pass

    elapsed = t1 - t0
    rtf = (elapsed / audio_sec) if audio_sec and audio_sec > 0 else None

    text = (result.get("text", "") or "").strip()
    if LOG_TRANSCRIPT:
        snippet = text.replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + " ..."
        print(f"[TRANSKRIPTION] {snippet}")

    print(
        f"[TIMING] Whisper {WHISPER_MODEL} tog {elapsed:.2f}s för {audio_sec or 'okänd'}s audio"
        f"{f' (RTF {rtf:.2f})' if rtf else ''}. Device={DEVICE}"
    )

    return text, {
        "whisper_model": WHISPER_MODEL,
        "device": DEVICE,
        "audio_seconds": audio_sec,
        "transcribe_seconds": round(elapsed, 3),
        "rtf": round(rtf, 3) if rtf else None
    }

# ------------------------ Transkribering (endpoint) ------------------------
@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    if request.method == "OPTIONS":
        return ("", 200)

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        fname = f"{int(time.time()*1000)}_{secure_filename(file.filename)}"
        filepath = os.path.join("/tmp", fname)
        file.save(filepath)
        try:
            size = os.path.getsize(filepath)
            print(f"[UPLOAD] Sparad till {filepath} ({size} bytes)")
        except Exception:
            print(f"[UPLOAD] Sparad till {filepath}")

        lang = request.form.get("language", LANGUAGE)
        text, meta = _transcribe_file(filepath, language=lang)
        return jsonify({"text": text, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------ Mistral 7B (sammanfattning) ------------------------
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
print(f"[BOOT] Laddar Mistral: {MISTRAL_MODEL_NAME}")

mistral_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
mistral_model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_NAME,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

PROTOKOLL_MALL = """Mötestitel: [kort titel baserad på innehållet]
Datum: [dagens datum om det framgår, annars lämna tomt]
Plats/Plattform: [ange om det framgår]
Deltagare: [ange om det framgår, annars skriv 'Ej specificerat']

§1. Sammanfattning
- [kort beskrivning av mötet och huvudpunkterna]

§2. Beslut
- [lista alla beslut tydligt i punktform]

§3. Åtgärder
- [lista åtgärder, ansvarig person och deadline om möjligt]

§4. Uppföljning
- [vad som ska följas upp, när och av vem]
"""

SYSTEM_PROMPT = (
    "Du skriver formella mötesprotokoll på korrekt svenska. "
    "Svara endast med det färdiga protokollet. Inga förklaringar, ingen extra text."
)

def summarize_text(transcript: str) -> str:
    user_prompt = (
        "Utgå från följande transkription och fyll i mallen korrekt. "
        "Behåll rubrikerna exakt som i mallen och ersätt klamrarna med innehåll. "
        "Svara ENDAST med det ifyllda protokollet.\n\n"
        f"Transkription:\n{transcript}\n\nMall:\n{PROTOKOLL_MALL}"
    )

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Skapa chat-input och generera enbart nya tokens
    input_ids = mistral_tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    # Flytta input till samma device som modellen använder
    model_device = next(mistral_model.parameters()).device
    input_ids = input_ids.to(model_device)

    t0 = time.time()
    out = mistral_model.generate(
        input_ids=input_ids,
        max_new_tokens=900,
        do_sample=False,
        temperature=0.0
    )
    t1 = time.time()

    gen = mistral_tokenizer.decode(
        out[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()

    print(f"[TIMING] Mistral-summering tog {t1 - t0:.2f}s")
    return gen

# ------------------------ Summering (endpoint) ------------------------
@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == "OPTIONS":
        return ("", 200)

    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text'"}), 400

        protocol = summarize_text(text)
        return jsonify({"protocol": protocol})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------ Healthcheck ------------------------
@app.route("/", methods=["GET"])
def index():
    return "Backend is running."

# ------------------------ Entrypoint ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
