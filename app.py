import os
from flask import Flask, request, render_template
import joblib
import preprocess
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

app = Flask(__name__)
MODELS_DIR = 'models'
# Load models
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
clf = joblib.load(os.path.join(MODELS_DIR, 'vgg16_classifier.pkl'))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
feat_model = Model(inputs=base_model.input, outputs=base_model.output)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

def analyze_audio_file(path):
    img = preprocess.extract_mfcc_spectrogram(path)
    if img is None:
        return None, "Error: failed to process audio"
    feats = feat_model.predict(img).flatten().reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    pred = clf.predict(feats_scaled)[0]
    prob = clf.predict_proba(feats_scaled)[0]
    return pred, prob

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return render_template('index.html', message='No file part')
        f = request.files['audio_file']
        if f.filename == '':
            return render_template('index.html', message='No selected file')
        if not f.filename.lower().endswith('.wav'):
            return render_template('index.html', message='Only .wav files allowed')

        save_path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(save_path)
        pred, prob = analyze_audio_file(save_path)
        os.remove(save_path)
        if pred is None:
            return render_template('index.html', message=prob)
        label = 'genuine' if pred == 0 else 'deepfake'
        confidence = float(prob[pred])
        return render_template('result.html', result=label, confidence=round(confidence, 4))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
