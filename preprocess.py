import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Parameters for spectrogram image size
IMG_SIZE = (224, 224)
TEMP_IMAGE = "temp_spectrogram.png"

def extract_mfcc_spectrogram(audio_path, n_mfcc=40, sr_target=None, save_path=TEMP_IMAGE):
    """Load audio, compute MFCC, render as image, and return a 224x224 RGB numpy array (scaled 0..1)."""
    try:
        y, sr = librosa.load(audio_path, sr=sr_target)
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Convert to dB for better visualization (optional)
        mfcc_db = librosa.power_to_db(np.abs(mfcc)) if mfcc.ndim else mfcc

        plt.figure(figsize=(2.24, 2.24), dpi=100)
        plt.axis('off')
        plt.imshow(mfcc_db, cmap='viridis', aspect='auto')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        image = tf.keras.preprocessing.image.load_img(save_path, target_size=IMG_SIZE)
        image = img_to_array(image) / 255.0
        # make sure 3-channels
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"[preprocess] Error processing {audio_path}: {e}")
        return None
