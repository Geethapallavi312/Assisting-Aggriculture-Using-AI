from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# ----------- Load Models and Resources -----------

# Crop Recommendation
with open('random_forest.pkl', 'rb') as f:
    crop_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    crop_scaler = pickle.load(f)
with open('crop_dict.pkl', 'rb') as f:
    crop_dict = pickle.load(f)

# Fertilizer Recommendation
fertilizer_model = pickle.load(open('classifier1.pkl', 'rb'))

fertilizer_classes = [
    'DAP', 'Fourteen-Thirty Five-Fourteen', 'Seventeen-Seventeen-Seventeen',
    'Ten-Twenty Six-Twenty Six', 'Twenty Eight-Twenty Eight', 'Twenty-Twenty', 'Urea'
]
fertilizer_encoder = LabelEncoder()
fertilizer_encoder.fit(fertilizer_classes)

fertilizer_scaler = StandardScaler()
fertilizer_scaler.mean_ = np.array([18.909091, 3.383838, 18.606061])
fertilizer_scaler.scale_ = np.array([11.599693, 5.814667, 13.476978])
fertilizer_scaler.var_ = fertilizer_scaler.scale_ ** 2
fertilizer_scaler.n_features_in_ = 3
fertilizer_scaler.n_samples_seen_ = 99

# Plant Disease Detection
disease_model = tf.keras.models.load_model('trained_model1.keras')
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ----------- Routes -----------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def crop_page():
    return render_template('Crop.html')

@app.route('/fertilizer')
def fertilizer_page():
    return render_template('fertilizers.html')

@app.route('/disese')
def disease_page():
    return render_template('disese.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



# ---- Crop Recommendation (JSON API) ----
@app.route('/get_crop_recommendation', methods=['POST'])
def recommend_crop():
    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled = crop_scaler.transform(features)
        prediction = crop_model.predict(scaled)[0]
        recommended_crop = crop_dict[prediction]

        return jsonify({'recommendation': recommended_crop})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ---- Fertilizer Recommendation (JSON API) ----
@app.route('/get_fertilizer_recommendation', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()
        N = float(data['nitrogen'])
        P = float(data['phosphorus'])
        K = float(data['potassium'])

        features = np.array([[N, K, P]])  # Note the order: N, K, P
        scaled = fertilizer_scaler.transform(features)
        prediction = fertilizer_model.predict(scaled)
        recommended_fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

        return jsonify({'result': f"🌾 {recommended_fertilizer}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ---- Plant Disease Detection (JSON API) ----
@app.route('/analyze', methods=['POST'])
def analyze_disease():
    try:
        if 'plantImage' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['plantImage']
        image = Image.open(file).convert('RGB')
        image = image.resize((128, 128))
        input_arr = np.expand_dims(np.array(image), axis=0)

        prediction = disease_model.predict(input_arr)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({'result': f"🦠 Predicted Disease: {predicted_class}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------- Run Server -----------

if __name__ == '__main__':
    app.run(debug=True)
