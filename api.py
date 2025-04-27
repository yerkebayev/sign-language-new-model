from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Load model and label map
with open('./model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
label_map = model_dict['label_map']  # label_map saved properly
labels_dict = {v: k for k, v in label_map.items()}  # Reverse map: id → label

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Load and preprocess image
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400

        data_aux = []
        x_total = []
        y_total = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_total.append(lm.x)
                y_total.append(lm.y)

        min_x = min(x_total)
        min_y = min(y_total)

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

        # Pad if only one hand detected
        if len(results.multi_hand_landmarks) == 1:
            for _ in range(21):
                data_aux.append(0.0)
                data_aux.append(0.0)

        # Check feature size
        data_aux = np.asarray(data_aux).reshape(1, -1)
        if data_aux.shape[1] != 84:
            return jsonify({'error': f'Invalid input size: {data_aux.shape[1]}, expected 84'}), 400

        # Predict
        probabilities = model.predict_proba(data_aux)
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100  # Convert to %

        predicted_label = labels_dict.get(predicted_idx, f"Unknown {predicted_idx}")

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Always run without debug mode for stable server
    app.run(host='0.0.0.0', port=5000, debug=False)
