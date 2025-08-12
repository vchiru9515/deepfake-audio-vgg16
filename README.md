# Deepfake Audio Detection (MFCC -> VGG16 -> SVM)

This repo shows how to extract MFCC spectrogram images from audio and use VGG16 as a feature extractor, followed by a lightweight SVM classifier for final prediction. A Flask app is included for inference.

## Quickstart
1. Install requirements: `pip install -r requirements.txt`
2. Download dataset: `python download_dataset.py` (configure dataset slug or manually place data in `data/genuine` and `data/deepfake`)
3. Train: `python train.py`
4. Run web app: `python app.py`

## Notes
- Large datasets should not be committed to GitHub. Use the provided download script to fetch data.
- The model pipeline can be swapped: train end-to-end CNN or fine-tune VGG16 if you have large labelled data.
