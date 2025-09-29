import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Tillåt frontend att prata med backend
CORS(app, origins=["https://irl-protokoll-frontend.onrender.com"])

# Skapa katalog för att spara uppladdade filer
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
    return jsonify({
        "status": "ok",
        "filename": filename,
        "saved_to": save_path
    })

if __name__ == "__main__":
    # Render sätter PORT i miljövariabler
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
