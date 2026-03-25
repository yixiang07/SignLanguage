"""
Real-Time ASL Fingerspelling → Text with GPT-2 Autocomplete
=============================================================
Runs the trained ResNet50 model on a live webcam feed. A stability
filter suppresses noise, and GPT-2 completes partial words on demand.

Controls:
    Enter     — show GPT-2 autocomplete suggestion
    s         — autocomplete and replace the buffer
    Space     — insert a space between words
    Backspace — delete the last character
    r         — reset everything
    Esc       — quit

Usage:
    python predict.py
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = "resnet_sign_language_best_model.h5"
CONSISTENCY_THRESHOLD = 10          # frames before a letter is accepted
GPT2_MAX_NEW_TOKENS = 5
PADDING_PX = 20

# Class index → ASL letter (24 static signs, J and Z excluded)
CLASS_MAP = {
    0: "A",  1: "B",  2: "K",  3: "L",  4: "M",  5: "N",  6: "O",
    7: "P",  8: "Q",  9: "R", 10: "S", 11: "T", 12: "C", 13: "U",
    14: "V", 15: "W", 16: "X", 17: "Y", 18: "D", 19: "E", 20: "F",
    21: "G", 22: "H", 23: "I",
}

# ── Load models ──────────────────────────────────────────────
print("Loading ResNet50 classifier...")
classifier = load_model(MODEL_PATH)

print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2.eval()

# ── MediaPipe hand detector ─────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def get_hand_roi(img):
    """Detect a hand and return the cropped ROI + bounding box."""
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None, None

    lm = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    x_min = max(int(min(xs) * w) - PADDING_PX, 0)
    y_min = max(int(min(ys) * h) - PADDING_PX, 0)
    x_max = min(int(max(xs) * w) + PADDING_PX, w)
    y_max = min(int(max(ys) * h) + PADDING_PX, h)

    return img[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def autocomplete(letter_buffer):
    """Use GPT-2 to complete the last word in the letter buffer."""
    raw = "".join(letter_buffer).strip()
    words = raw.split()
    if not words:
        return ""

    *prefix, current = words
    prompt = " ".join(prefix) + (" " if prefix else "") + current
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = gpt2.generate(
            input_ids,
            max_new_tokens=GPT2_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result_words = tokenizer.decode(output_ids[0], skip_special_tokens=True).split()

    if len(result_words) <= len(prefix):
        return raw

    completed_word = result_words[len(prefix)]
    return " ".join(prefix + [completed_word])


# ── Main loop ────────────────────────────────────────────────
def main():
    letter_buffer = []
    current_pred = ""
    pred_count = 0
    gpt2_output = ""

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        roi, bbox = get_hand_roi(frame)
        predicted_letter = ""

        if roi is not None:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224)).astype("float32")
            processed = np.expand_dims(preprocess_input(resized), axis=0)

            pred = classifier.predict(processed)
            predicted_letter = CLASS_MAP.get(int(np.argmax(pred)), "")

        # ── Stability filter ─────────────────────────────────
        if predicted_letter:
            if predicted_letter == current_pred:
                pred_count += 1
            else:
                current_pred = predicted_letter
                pred_count = 1

            if pred_count >= CONSISTENCY_THRESHOLD:
                letter_buffer.append(current_pred)
                pred_count = 0
                current_pred = ""
        else:
            current_pred = ""
            pred_count = 0

        # ── Draw UI ──────────────────────────────────────────
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 2)

        panel = np.zeros((150, frame.shape[1], 3), dtype="uint8")
        cv2.rectangle(panel, (10, 10), (frame.shape[1] - 10, 140), (30, 30, 30), -1)
        cv2.rectangle(panel, (10, 10), (frame.shape[1] - 10, 140), (80, 80, 80), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "You Signed:", (30, 50),
                    font, 0.8, (100, 255, 200), 2)
        cv2.putText(panel, "".join(letter_buffer), (200, 50),
                    font, 1.2, (255, 255, 255), 2)
        cv2.putText(panel, "GPT-2 Suggestion:", (30, 110),
                    font, 0.8, (255, 255, 100), 2)
        cv2.putText(panel, gpt2_output, (330, 110),
                    font, 1.2, (255, 255, 255), 2)

        cv2.imshow("ASL Fingerspelling + GPT-2", np.vstack([frame, panel]))

        # ── Keyboard controls ────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                               # Esc
            break
        elif key == ord("r"):                        # Reset
            letter_buffer, gpt2_output = [], ""
            current_pred, pred_count = "", 0
        elif key == 13:                              # Enter → suggest
            gpt2_output = autocomplete(letter_buffer) if letter_buffer else "[No input]"
        elif key == ord(" "):                        # Space
            letter_buffer.append(" ")
        elif key == 8 and letter_buffer:             # Backspace
            letter_buffer.pop()
        elif key == ord("s") and letter_buffer:      # s → autocomplete + replace
            letter_buffer = list(autocomplete(letter_buffer))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
