import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import preprocess

# Config
DATA_DIR = "data"
GENUINE_DIR = os.path.join(DATA_DIR, "genuine")
DEEPFAKE_DIR = os.path.join(DATA_DIR, "deepfake")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load VGG16 base (imagenet weights) as feature extractor
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feat_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_folder(folder):
    X = []
    files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        img = preprocess.extract_mfcc_spectrogram(path)
        if img is None:
            continue
        feats = feat_model.predict(img)
        X.append(feats.flatten())
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(files)} in {folder}")
    return np.array(X)

def main():
    print("Extracting features for genuine samples...")
    X_genuine = extract_features_from_folder(GENUINE_DIR)
    print("Extracting features for deepfake samples...")
    X_deepfake = extract_features_from_folder(DEEPFAKE_DIR)

    # Labels
    y_genuine = np.zeros(len(X_genuine), dtype=int)
    y_deepfake = np.ones(len(X_deepfake), dtype=int)

    X = np.vstack([X_genuine, X_deepfake])
    y = np.concatenate([y_genuine, y_deepfake])

    print("Feature shape:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier (SVM)
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save models
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(clf, os.path.join(MODELS_DIR, 'vgg16_classifier.pkl'))
    print("Saved scaler and classifier to models/")

if __name__ == '__main__':
    main()
