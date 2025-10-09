# app.py — version med långfilsstöd och förberett live-läge

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import math
import tempfile

# ------------------------ Flask + CORS ------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Tillåt upp till 500 MB ljud

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
LOG_TRANSCRIPT = os.getenv("LOG_TRANSCRIPT", "0") == "1"
LANGUAGE = os.getenv("LANGUAGE", "sv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = (DEVICE == "cuda")

# ------------------------ Långfilshantering ------------------------
def split_audio_if_needed(input_path, max_minutes=10):
    """Delar upp långa ljudfiler i bitar (varje max_minutes minuter)."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ]
        duration = float(subprocess.check_output(cmd).decode().strip())
        if duration <= max_minutes * 60:
            return [input_path]
        print(f"[SPLIT] Ljudet är {duration/60:.1f} min långt — delar upp...")
        segments = []
        num_parts = math.ceil(duration / (max_minutes * 60))
        for i in range(num_parts):
            start = i * max_minutes * 60
            output_part = f"{input_path}_part{i}.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, "-ss", str(start),
                "-t", str(max_minutes * 60), "-c", "copy", output_part
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segments.append(output_part)
        return segments
    except Exception as e:
        print(f"[SPLIT ERROR] {e}")
        return [input_path]

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
    try:
        audio_sec = float(result.get("segments", [{}])[-1].get("end", 0))
    except Exception:
        audio_sec = None

    elapsed = t1 - t0
    rtf = (elapsed / audio_sec) if audio_sec and audio_sec > 0 else None
    text = (result.get("text", "") or "").strip()
    if LOG_TRANSCRIPT:
        snippet = text.replace("\n", " ")
        print(f"[TRANSKRIPTION] {snippet[:240]}{' ...' if len(snippet)>240 else ''}")
    print(f"[TIMING] Whisper {WHISPER_MODEL} tog {elapsed:.2f}s för {audio_sec or 'okänd'}s audio"
          f"{f' (RTF {rtf:.2f})' if rtf else ''}. Device={DEVICE}")
    return text

# ------------------------ Transkribering (endpoint) ------------------------
@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    if request.method == "OPTIONS":
        return ("", 200)
    try:
        # live=1 → buffrad realtidsinspelning
        live_mode = request.args.get("live", "0") == "1"

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # skapa temporärt namn
        fname = f"{int(time.time()*1000)}_{secure_filename(file.filename)}"
        filepath = os.path.join("/tmp", fname)
        file.save(filepath)
        print(f"[UPLOAD] {filepath} mottagen (live={live_mode})")

        lang = request.form.get("language", LANGUAGE)

        if live_mode:
            # enklare hantering: direkt transkribera hela blocket
            text = _transcribe_file(filepath, language=lang)
            return jsonify({"partial_text": text})
        else:
            # vanlig helfilstranskribering
            segments = split_audio_if_needed(filepath)
            all_text = []
            for part in segments:
                t = _transcribe_file(part, language=lang)
                all_text.append(t)
            return jsonify({"text": "\n".join(all_text)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------ Mistral 7B ------------------------
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
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Utgå från följande transkription och fyll i mallen korrekt. "
            "Behåll rubrikerna exakt som i mallen och ersätt klamrarna med innehåll. "
            "Svara ENDAST med det ifyllda protokollet.\n\n"
            f"Transkription:\n{transcript}\n\nMall:\n{PROTOKOLL_MALL}"}
    ]
    input_ids = mistral_tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    input_ids = input_ids.to(next(mistral_model.parameters()).device)
    t0 = time.time()
    out = mistral_model.generate(input_ids=input_ids, max_new_tokens=900, do_sample=False, temperature=0.0)
    gen = mistral_tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"[TIMING] Mistral-summering tog {time.time() - t0:.2f}s")
    return gen

# ------------------------ Summering ------------------------
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
