import os
from flask import Flask, request, jsonify

app = Flask(__name__)

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
    return jsonify({"status": "ok", "filename": filename, "saved_to": save_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

