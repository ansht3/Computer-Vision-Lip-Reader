# Computer Vision Lip Reader

Real-time lip reading using a 3D CNN. It detects a face, tracks the mouth region, and predicts a word from short video clips.

## What it does

- Collects short clips of your mouth while you speak a word
- Preprocesses clips into fixed-length frame sequences
- Trains a small 3D CNN classifier
- Runs live webcam prediction with confidence scores

## Features

- Real-time mouth tracking (dlib + OpenCV)
- 3D CNN for spatiotemporal features
- Simple data collection + training pipeline
- Multi-class word classification with confidences

## Requirements

- Python 3.8+
- Webcam
- Good lighting + a clear view of your face
- dlib landmark model: `shape_predictor_68_face_landmarks.dat`

## Setup

```bash
git clone <repository-url>
cd LipReadVox

python -m venv lipread_env
source lipread_env/bin/activate   # (Windows: lipread_env\Scripts\activate)

pip install -r requirements.txt
```

Download the dlib landmark predictor:

```bash
curl -L -o model/shape_predictor_68_face_landmarks.dat.bz2   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
cd model && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && cd ..
```

## Quick demo

This runs an end-to-end flow (sample data → preprocess → train → test):

```bash
python quick_demo.py
```

## Run live prediction

```bash
python src/predict.py
```

Press **Q** to quit.

## Collect your own data

Use the demo collector:

```bash
python demo_collection.py
```

Controls:
- **R** start recording
- **S** stop recording
- **Q** quit

Tips:
- Record **3–5 takes per word** to start (more helps a lot).
- Keep your head distance consistent.
- Avoid harsh shadows across your mouth.

## Train on your words

After collecting data, run:

```bash
python src/preprocess_training.py
python src/model_training.py
```

Training generates:
- `model/lip_reader_3dcnn.h5`
- `model/labels.npy`

Then run live prediction again:

```bash
python src/predict.py
```

## Notes on the model

- Input: **22 frames** of **80×112** grayscale mouth crops
- 3D conv blocks + dropout + global pooling + dense classifier
- Accuracy varies with your dataset and recording consistency

## Troubleshooting

**Camera not found**
- Check camera permissions
- Try changing the camera index in the code

**Face / mouth not detected**
- Improve lighting and make your face more centered
- Confirm the dlib file exists:
  `model/shape_predictor_68_face_landmarks.dat`

**Low accuracy**
- Collect more examples per word
- Keep the same framing/lighting between training and demo
- Start with fewer, visually distinct words (e.g., “yes”, “no”, “hello”)

**Training runs out of memory**
- Reduce batch size
- Reduce model size or sequence length

## Future Improvements

- Add more speakers / lighting conditions for better generalization
- Package as a small app (CLI + saved configs) for easier reuse
