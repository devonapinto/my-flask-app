from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS  # Add this
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = load_model('1.keras')

# Load class names (replace this with your own class names)
class_names = ['pepper_bell_bacterial_spot', 'pepper_bell_healthy', 'tomato_bacterial_spot','tomato_early_blight','tomato_healthy','tomato_late_blight']


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Save the uploaded image
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]

    # Clean up
    os.remove(filepath)

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Ensure the upload folder exists
    app.run(debug=True, host='0.0.0.0', port=5000)

