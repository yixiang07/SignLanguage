# Sign Language to Text — Real-Time ASL Fingerspelling with GPT-2 Autocomplete

A computer vision pipeline that translates American Sign Language (ASL) fingerspelling into text in real time, using a fine-tuned ResNet50 classifier and GPT-2 word completion to bridge the communication gap for the deaf and hard-of-hearing community.

## Demo

The system runs three stages in a single webcam loop:

1. **Hand detection** — MediaPipe isolates the hand region from the video feed.
2. **Letter classification** — A fine-tuned ResNet50 predicts the signed letter (24 static ASL letters, excluding J and Z which require motion).
3. **Word completion** — Pressing Enter sends the accumulated letter buffer to GPT-2, which autocompletes the current word to accelerate communication.

## Pipeline

```
Webcam → MediaPipe Hand ROI → ResNet50 (24-class) → Stability Filter → Letter Buffer → GPT-2 Autocomplete
```

## Repository Structure

```
├── collect_images.py       # Capture training images via webcam + MediaPipe
├── train.py                # Fine-tune ResNet50 with KerasTuner hyperparameter search
├── predict.py              # Real-time inference with GPT-2 word completion
├── requirements.txt
└── .gitignore
```

## How It Works

### Data Collection (`collect_images.py`)
Captures hand images from a webcam using MediaPipe for hand detection. Each frame is cropped to the hand bounding box, converted to grayscale, resized to 224×224, and saved to a class-specific folder under `dataset/`. Random horizontal flips are applied during capture for data augmentation.

**Controls:** Press `c` to start/stop capturing, `Esc` to exit.

### Model Training (`train.py`)
Fine-tunes a pretrained ResNet50 (ImageNet weights) on the collected dataset:

- Automatically splits raw data into 70/15/15 train/val/test sets.
- Applies augmentation (rotation, zoom, shift, flip) during training.
- Uses **KerasTuner RandomSearch** to optimise the unfreezing depth, dense layer size, dropout rate, and learning rate.
- Saves the best model to `resnet_sign_language_best_model.h5`.

### Real-Time Prediction (`predict.py`)
Runs the trained model on a live webcam feed:

- **Stability filter** — A letter is only appended to the buffer after it has been predicted consistently for 10 consecutive frames, reducing noise from transient misclassifications.
- **GPT-2 autocomplete** — On keypress, the accumulated letters are sent to GPT-2 which completes the current word using greedy decoding.
- **Keyboard controls:**
  - `Enter` — autocomplete the current word (displayed as a suggestion)
  - `s` — autocomplete and replace the buffer with the completed word
  - `Space` — insert a space between words
  - `Backspace` — delete the last character
  - `r` — reset the buffer
  - `Esc` — quit

## Supported Letters

The model classifies 24 static ASL fingerspelling signs:

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

J and Z are excluded as they require motion trajectories that a single-frame classifier cannot capture.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Collect training data
```bash
python collect_images.py
# Enter a class number (0–23) when prompted, then press 'c' to start capturing
# Repeat for each letter
```

### 2. Train the model
```bash
python train.py
```

### 3. Run real-time prediction
```bash
python predict.py
```

## Tech Stack

- **TensorFlow / Keras** — ResNet50 transfer learning and KerasTuner hyperparameter optimisation
- **MediaPipe** — Real-time hand detection and landmark tracking
- **OpenCV** — Webcam capture and UI rendering
- **Hugging Face Transformers** — GPT-2 for word-level autocomplete
- **PyTorch** — GPT-2 inference runtime
