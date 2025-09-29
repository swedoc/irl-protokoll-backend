import os
import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://irl-protokoll-frontend.onrender.com"])

# Ladda modellen en gång vid start
model = whisper.load_model("tiny")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil skickades"}), 400

    f = request.files["file"]
    filename = f.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    # Kör transkribering
    try:
        result = model.transcribe(save_path, language="sv")
        text = result.get("text", "").strip()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({
        "status": "ok",
        "filename": filename,
        "transcription": text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

