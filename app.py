from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil skickades"}), 400
    f = request.files["file"]
    filename = f.filename
    # just nu sparar vi inte filen, bara kvitterar
    return jsonify({"status": "ok", "filename": filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

