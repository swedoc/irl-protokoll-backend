import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app, origins=["https://irl-protokoll-frontend.onrender.com"])

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ladda modellen en gång (tiny = snabbast, CPU)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil skickades"}), 400
    f = request.files["file"]
    filename = f.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    # kör transkribering
    segments, info = model.transcribe(save_path, beam_size=1)
    text = " ".join([seg.text for seg in segments])

    return jsonify({
        "status": "ok",
        "filename": filename,
        "language": info.language,
        "transcription": text.strip()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
