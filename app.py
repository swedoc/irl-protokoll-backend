from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Ladda Huggingface Whisper pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    # Spara temporärt
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    # Transkribera
    result = pipe(filepath)
    text = result["text"]

    return jsonify({"text": text})

@app.route("/", methods=["GET"])
def index():
    return "Backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
