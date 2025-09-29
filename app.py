import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Tillåt endast din frontend på Render
FRONTEND_ORIGIN = "https://irl-protokoll-frontend.onrender.com"
CORS(app, resources={r"/upload": {"origins": FRONTEND_ORIGIN}})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil skickades"}), 400
    f = request.files["file"]
    filename = f.filename or "okänt_namn"
    # Här sparar vi ännu inte. Vi kvitterar bara.
    return jsonify({"status": "ok", "filename": filename})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

