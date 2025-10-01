from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
CORS(app)

# ----------- Transkribering -----------
# HuggingFace Whisper (byt modellnamn här: tiny → medium)
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    result = pipe(filepath)
    text = result["text"]

    return jsonify({"text": text})


# ----------- Summering / protokoll -----------
# Ladda Mistral 7B (Instruct)
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

device = "cuda" if torch.cuda.is_available() else "cpu"

mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_model_id,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id)

def summarize_text(text: str) -> str:
    prompt = f"Gör ett neutralt, strukturerat mötesprotokoll av följande text:\n\n{text}\n\n"
    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mistral_model.generate(**inputs, max_new_tokens=500)
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
