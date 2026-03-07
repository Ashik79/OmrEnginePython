from flask import Flask, request, jsonify
from flask_cors import CORS
from omr_engine import OmrEngine
import cv2
import numpy as np
import base64

app = Flask(__name__)
# Enable CORS so the React app can connect
CORS(app)

engine = OmrEngine()

@app.route('/scan', methods=['POST'])
def scan_omr():
    """
    POST /scan
    Body JSON:
      {
        "image": "<base64 image data>",
        "active_q": 25        ← optional, default 75 (how many Q to evaluate)
      }
    
    UNIVERSAL SHEET: The printed OMR sheet always has 75 bubbles.
    active_q tells the engine how many to evaluate, rest are SKIPPED_INACTIVE.
    """
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Get base64 image data
        image_data = data['image']
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        # Get active_q — default to 60 (evaluate all)
        active_q = int(data.get('active_q', 60))
        active_q = max(5, min(60, active_q))   # clamp to valid range

        # Decode base64 to numpy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Save temporary file for the engine
        temp_filename = "temp_scan.jpg"
        cv2.imwrite(temp_filename, img)

        # Run OMR Engine with active_q parameter
        result = engine.run(temp_filename, active_q=active_q)

        return jsonify({
            "success": True,
            "result": result,
            "active_q": active_q,
            "total_sheet_q": 60
        })

    except Exception as e:
        print("Error processing image:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "engine": "Universal 75-Q OMR", "version": "2.0"})

if __name__ == '__main__':
    print("=" * 50)
    print("  Python OMR Server - Universal 75-Q Sheet")
    print("  Port: 5050")
    print("  POST /scan  ->  { image, active_q }")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5050, debug=True)
