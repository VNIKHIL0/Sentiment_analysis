from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from deepface import DeepFace

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_data = data['image'].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except:
        emotion = "No Face Detected"

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
