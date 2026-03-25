"""
Collect Training Images
=======================
Captures hand images from a webcam using MediaPipe for detection.
Each frame is cropped to the hand bounding box, converted to grayscale,
resized to 224×224, and saved to `dataset/<class_id>/`.

Controls:
    c   — start / stop capturing
    Esc — exit

Usage:
    python collect_images.py
    # Enter a class number (0–23) when prompted
"""

import os
import random
import cv2
import mediapipe as mp

# ── Configuration ────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
DATASET_DIR = "dataset"
IMAGES_PER_CLASS = 400
PADDING_PX = 20

# ── MediaPipe hand detector ─────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def get_hand_roi(img):
    """Detect a hand via MediaPipe and return the cropped ROI + bounding box."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    landmarks = results.multi_hand_landmarks[0]
    h, w, _ = img.shape

    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]

    x_min = max(int(min(x_coords) * w) - PADDING_PX, 0)
    y_min = max(int(min(y_coords) * h) - PADDING_PX, 0)
    x_max = min(int(max(x_coords) * w) + PADDING_PX, w)
    y_max = min(int(max(y_coords) * h) + PADDING_PX, h)

    roi = img[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)
    return roi, bbox


def store_images(class_id):
    """Capture and save hand images for a single gesture class."""
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)

    folder = os.path.join(DATASET_DIR, str(class_id))
    os.makedirs(folder, exist_ok=True)

    pic_no = 0
    capturing = False
    warmup_frames = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        roi, bbox = get_hand_roi(frame)

        # Save images only after a short stabilisation period
        if roi is not None and warmup_frames > 10 and capturing:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            save_img = cv2.resize(gray_roi, IMAGE_SIZE)

            # Random horizontal flip for augmentation
            if random.randint(0, 1):
                save_img = cv2.flip(save_img, 1)

            pic_no += 1
            cv2.imwrite(os.path.join(folder, f"{pic_no}.jpg"), save_img)
            cv2.putText(frame, "Capturing...", (30, 60),
                        cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255), 2)

        if bbox is not None:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {pic_no}", (30, 400),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255), 2)
        cv2.imshow("Capturing gesture", frame)

        if roi is not None:
            cv2.imshow("Hand ROI", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            capturing = not capturing
            warmup_frames = 0
        if capturing:
            warmup_frames += 1
        if key == 27 or pic_no >= IMAGES_PER_CLASS:
            break

    cam.release()
    cv2.destroyAllWindows()


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    class_id = input("Enter gesture number (0–23): ")
    store_images(class_id)
