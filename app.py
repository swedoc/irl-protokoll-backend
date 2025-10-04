from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ----------- Transkribering (Whisper på GPU) -----------
whisper_model = whisper.load_model("medium")  # välj small / medium / large-v3 beroende på behov

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    result = whisper_model.transcribe(filepath)
    text = result["text"]

    return jsonify({"text": text})


# ----------- Summering / protokoll (Mistral 7B) -----------
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

device = "cuda" if torch.cuda.is_available() else "cpu"

mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_id,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)

def summarize_text(text: str) -> str:
    prompt = f"""Sammanfatta följande transkription som ett formellt mötesprotokoll.
Använd denna struktur exakt:

Mötestitel: [kort titel baserad på innehållet]
Datum: [dagens datum om det framgår, annars lämna tomt]
Plats/Plattform: [ange om det framgår]
Deltagare: [ange om det framgår, annars skriv 'Ej specificerat']

§1. Sammanfattning
- Kort beskrivning av mötet och huvudpunkterna.

§2. Beslut
- Lista tydligt alla beslut som framgår.

§3. Åtgärder
- Lista tydligt alla åtgärder, med ansvarig om det går att utläsa.

§4. Nästa steg
- Vad som planerats framåt, nästa möte etc.

Transkription:
{text}
"""
    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mistral_model.generate(**inputs, max_new_tokens=600)
    return mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400
    
    text = data["text"]
    protocol = summarize_text(text)
    return jsonify({"protocol": protocol})


# ----------- Healthcheck -----------
@app.route("/", methods=["GET"])
def index():
    return "Backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
