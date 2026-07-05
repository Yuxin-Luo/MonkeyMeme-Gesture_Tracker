"""
Gesture Tracker — MediaPipe Tasks API edition.

Detects hand & face landmarks through the webcam using MediaPipe 0.10+ Tasks API
(HandLandmarker + FaceLandmarker), classifies gestures (Thumbs Up, Pointing,
Thinking, Neutral), and pairs each detection with the corresponding meme image.

First run: if the .task model files are missing under ./models/, they are
downloaded automatically from the official MediaPipe model registry.
"""

import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Tasks setup
# ---------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Drawing helpers — these live under mp.tasks.vision.drawing_utils in 0.10+.
mp_drawing = mp.tasks.vision.drawing_utils
HandConnections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# Hand landmark indices — same topology as the legacy FaceMesh/Hands solutions.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20
NOSE_TIP = 4  # Index 4 in FaceMesh corresponds to the nose tip.

# Tolerance for "tip is clearly below PIP" — small forgiving margin (in
# normalized image coordinates) so a half-curled finger still counts as curled.
CURL_TOLERANCE = 0.0

# Display palette (BGR) — one color per recognised gesture, used to draw the
# large text label in the middle of the camera panel.
GESTURE_COLORS = {
    "THUMBS_UP":  ( 50, 200,  50),  # green   — positive
    "THUMBS_DOWN":( 50,  50, 220),  # red     — negative
    "POINTING":   (220, 130,  50),  # blue
    "PEACE":      ( 50, 220, 220),  # yellow  (BGR)
    "OPEN_PALM":  (220, 220,  50),  # cyan
    "FIST":       (150, 150, 150),  # gray
    "THINKING":   (220,  50, 220),  # magenta
    "NEUTRAL":    (200, 200, 200),  # light gray
}

# Model files & download URLs (Google's official MediaPipe model registry).
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_models():
    """Download the .task models on first run, no-op if already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, url in ((HAND_MODEL_PATH, HAND_MODEL_URL),
                      (FACE_MODEL_PATH, FACE_MODEL_URL)):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        print(f"Downloading model: {os.path.basename(path)} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:  # noqa: BLE001 — surface a clear actionable error.
            raise RuntimeError(
                f"Failed to download {url} -> {path}: {exc}\n"
                "Place hand_landmarker.task and face_landmarker.task manually "
                f"inside {MODELS_DIR} if your network blocks the download."
            ) from exc
        print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Gesture images
# ---------------------------------------------------------------------------

IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "NEUTRAL": "neutral.jpg",
    "THINKING": "thinking.jpg",
}


def load_and_resize_image(path, target_height):
    """Load an image and resize it to match the camera frame height."""
    full_path = os.path.join(os.getcwd(), path)
    img = cv2.imread(full_path)
    if img is None:
        print(f"Error: Failed to load image {path}. "
              "Make sure the file exists in the same directory.")
        return None

    ratio = target_height / img.shape[0]
    target_width = int(img.shape[1] * ratio)
    return cv2.resize(img, (target_width, target_height))


# ---------------------------------------------------------------------------
# Gesture classification
# ---------------------------------------------------------------------------

def _finger_curled(landmarks, pip_idx, tip_idx, tolerance=CURL_TOLERANCE):
    """A finger is curled when its tip sits below its own PIP joint (in image-y).

    This is checked per-finger — using each finger's own PIP joint as the
    reference, which is what the previous version of this script got wrong.
    """
    return landmarks[tip_idx].y > landmarks[pip_idx].y + tolerance


def _finger_extended(landmarks, pip_idx, tip_idx, tolerance=CURL_TOLERANCE):
    """A finger is extended when its tip sits above its own PIP joint."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y - tolerance


def _middle_ring_pinky_curled(landmarks):
    """Middle / ring / pinky all curled into the palm — used by both branches
    because those three fingers should be folded in BOTH thumbs-up and pointing.
    The index is treated separately since it is the discriminating finger.
    """
    return all(
        _finger_curled(landmarks, pip_idx, tip_idx)
        for pip_idx, tip_idx in (
            (MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
            (RING_FINGER_PIP, RING_FINGER_TIP),
            (PINKY_PIP, PINKY_TIP),
        )
    )


def classify_gesture(hand_landmarks):
    """Classify a wide range of static hand gestures.

    Returns one of:
      THUMBS_UP, THUMBS_DOWN, POINTING, PEACE, OPEN_PALM, FIST, NEUTRAL.

    Decision order (strongest signal first):
      1. THUMBS_DOWN  — thumb points clearly downward
      2. THUMBS_UP    — thumb is clearly the topmost finger
      3. FIST         — 4 fingers curled, no strong thumb signal
      4. OPEN_PALM    — all 4 fingers extended
      5. PEACE        — index + middle extended, ring + pinky curled
      6. POINTING     — index extended, others curled
      7. NEUTRAL      — anything else

    Tuning notes
    ------------
    THUMBS_DOWN does NOT require index_curled — real users often hold their
    index loosely while making a thumbs-down, and forcing it curled made the
    gesture fall through to POINTING (because the extended index then became
    the topmost finger).
    THUMBS_UP requires the thumb tip to sit a meaningful margin above the
    index tip (THUMBS_UP_GAP). Without this margin, a fist in which the thumb
    rests slightly on top of the curled fingers would misfire as THUMBS_UP
    because `tip < pip < mcp` is also true for a horizontal-ish thumb.
    """
    # Per-finger extended state (tip above the finger's own PIP AND MCP).
    index_extended_up = (
        hand_landmarks[INDEX_FINGER_TIP].y < hand_landmarks[INDEX_FINGER_PIP].y
        and hand_landmarks[INDEX_FINGER_TIP].y
        < hand_landmarks[INDEX_FINGER_MCP].y
    )
    middle_extended_up = (
        hand_landmarks[MIDDLE_FINGER_TIP].y < hand_landmarks[MIDDLE_FINGER_PIP].y
        and hand_landmarks[MIDDLE_FINGER_TIP].y
        < hand_landmarks[MIDDLE_FINGER_MCP].y
    )
    ring_extended_up = (
        hand_landmarks[RING_FINGER_TIP].y < hand_landmarks[RING_FINGER_PIP].y
        and hand_landmarks[RING_FINGER_TIP].y
        < hand_landmarks[RING_FINGER_MCP].y
    )
    pinky_extended_up = (
        hand_landmarks[PINKY_TIP].y < hand_landmarks[PINKY_PIP].y
        and hand_landmarks[PINKY_TIP].y < hand_landmarks[PINKY_MCP].y
    )

    middle_ring_pinky_curled = (
        not middle_extended_up
        and not ring_extended_up
        and not pinky_extended_up
    )
    index_curled = _finger_curled(hand_landmarks,
                                  INDEX_FINGER_PIP, INDEX_FINGER_TIP)

    # Thumb state — measured against its own joints plus the index tip / wrist.
    thumb_tip_y = hand_landmarks[THUMB_TIP].y
    thumb_ip_y = hand_landmarks[THUMB_IP].y
    thumb_mcp_y = hand_landmarks[THUMB_MCP].y
    index_tip_y = hand_landmarks[INDEX_FINGER_TIP].y
    wrist_y = hand_landmarks[WRIST].y

    # Thumb pointing DOWN: tip clearly below both wrist AND thumb MCP.
    THUMB_DOWN_TOLERANCE = 0.02
    thumb_pointing_down = (
        thumb_tip_y > wrist_y + THUMB_DOWN_TOLERANCE
        and thumb_tip_y > thumb_mcp_y + THUMB_DOWN_TOLERANCE
    )

    # Thumb clearly the topmost finger: tip significantly above its own IP/MCP
    # AND a meaningful margin above the index tip. This prevents a fist with
    # the thumb slightly resting on top from being misread as THUMBS_UP.
    THUMBS_UP_TOLERANCE = 0.03
    THUMBS_UP_GAP = 0.10
    thumb_topmost_significantly = (
        thumb_tip_y < thumb_ip_y - THUMBS_UP_TOLERANCE
        and thumb_tip_y < thumb_mcp_y - THUMBS_UP_TOLERANCE
        and thumb_tip_y < index_tip_y - THUMBS_UP_GAP
    )

    # 1) THUMBS_DOWN — strong directional signal; index state does not matter.
    if thumb_pointing_down and middle_ring_pinky_curled:
        return "THUMBS_DOWN"

    # 2) THUMBS_UP — thumb is clearly the topmost finger; index may be
    #    curled or slightly relaxed.
    if thumb_topmost_significantly and middle_ring_pinky_curled:
        return "THUMBS_UP"

    # 3) FIST — 4 fingers curled, no strong thumb signal.
    if middle_ring_pinky_curled and index_curled:
        return "FIST"

    # 4) OPEN_PALM — every non-thumb finger extended (thumb position irrelevant).
    if (index_extended_up and middle_extended_up
            and ring_extended_up and pinky_extended_up):
        return "OPEN_PALM"

    # 5) PEACE (V sign): index + middle extended, ring + pinky curled.
    if (index_extended_up and middle_extended_up
            and not ring_extended_up and not pinky_extended_up):
        return "PEACE"

    # 6) POINTING — index extended, middle/ring/pinky curled, thumb not on top.
    if index_extended_up and middle_ring_pinky_curled:
        return "POINTING"

    # 7) Anything else.
    return "NEUTRAL"


def check_thinking_gesture(hand_landmarks, face_landmarks,
                           frame_width, frame_height):
    """Whether the index fingertip is near the nose (Thinking gesture)."""
    if not hand_landmarks or not face_landmarks:
        return False

    index_tip = hand_landmarks[INDEX_FINGER_TIP]
    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    nose_tip = face_landmarks[NOSE_TIP]
    nose_x = int(nose_tip.x * frame_width)
    nose_y = int(nose_tip.y * frame_height)

    distance = np.sqrt((index_x - nose_x) ** 2 + (index_y - nose_y) ** 2)
    # Threshold scales with frame width — ≈8% (≈50 px on a 640-wide frame).
    # The previous 6% was too tight and caused flickering between THINKING
    # and POINTING at the boundary.
    MAX_DISTANCE = int(frame_width * 0.08)

    # Require the index tip to sit above the index MCP — this allows both a
    # fully extended index and a hook-shaped "thinking" curl, while still
    # rejecting a closed fist held next to the face (where the tip would be
    # below MCP).
    is_index_above_mcp = (
        hand_landmarks[INDEX_FINGER_TIP].y < hand_landmarks[INDEX_FINGER_MCP].y
    )

    return distance < MAX_DISTANCE and is_index_above_mcp


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hand_landmarks(image, hand_landmarks):
    """Draw hand landmarks using the Tasks-API drawing_utils."""
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        HandConnections,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
    )


def draw_gesture_label(image, gesture, frame_width, frame_height):
    """Draw the gesture name as large colored text in the center of the camera panel.

    A semi-transparent black rectangle is drawn behind the text so it stays
    readable regardless of the camera background. The text color comes from
    GESTURE_COLORS, so each gesture appears in its own distinct color.
    """
    text = gesture.replace("_", " ")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6
    thickness = 3

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Center horizontally in the camera panel, vertically around 1/3 from top.
    text_x = (frame_width - text_w) // 2
    text_y = frame_height // 3 + text_h // 2

    # Semi-transparent black box behind the text for readability.
    pad = 16
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (text_x - pad, text_y - text_h - pad),
        (text_x + text_w + pad, text_y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    color = GESTURE_COLORS.get(gesture, (255, 255, 255))
    cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

class GestureSmoother:
    """Smooth per-frame gesture labels using a sliding-window majority vote.

    Keeps the last `window_size` raw labels and adopts the most frequent one
    once it appears in at least `min_count` of those frames. This is more
    forgiving than a strict consecutive-frame debouncer: a few frames of
    model noise (or a borderline THINKING/POINTING flicker near the face)
    no longer flip the displayed label.

    Defaults: window_size=6, min_count=4 (≈67% majority). At 30 fps the user
    needs to hold a gesture for roughly 130 ms (4 frames) before it sticks,
    and up to 2 of every 6 frames can be wrong without losing the label.
    """

    def __init__(self, window_size=6, min_count=4):
        from collections import deque
        self.window_size = window_size
        self.min_count = min_count
        self.history = deque(maxlen=window_size)
        self.current = "NEUTRAL"

    def update(self, raw_gesture):
        self.history.append(raw_gesture)
        if not self.history:
            return self.current

        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
        best_gesture, best_count = max(counts.items(), key=lambda item: item[1])

        if best_count >= self.min_count:
            self.current = best_gesture
        return self.current


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_models()

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"Could not open camera at index {CAMERA_INDEX}.")

    print("Gesture Tracker running. Press 'q' to quit.")

    # VIDEO mode requires monotonically-increasing timestamps in milliseconds.
    start_time_ms = int(time.monotonic() * 1000)
    smoother = GestureSmoother(window_size=6, min_count=4)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.monotonic() * 1000) - start_time_ms
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = (
                hand_result.hand_landmarks[0] if hand_result.hand_landmarks else None
            )
            face_landmarks = (
                face_result.face_landmarks[0] if face_result.face_landmarks else None
            )

            raw_gesture = "NEUTRAL"
            if hand_landmarks:
                if hand_landmarks and face_landmarks:
                    if check_thinking_gesture(hand_landmarks, face_landmarks,
                                              frame_width, frame_height):
                        raw_gesture = "THINKING"

                if raw_gesture == "NEUTRAL":
                    raw_gesture = classify_gesture(hand_landmarks)

                draw_hand_landmarks(frame, hand_landmarks)

            current_gesture = smoother.update(raw_gesture)

            # Always show the gesture name as large colored text in the middle
            # of the camera panel — covers gestures that do and don't have
            # a paired meme image.
            draw_gesture_label(frame, current_gesture, frame_width, frame_height)

            if current_gesture in IMAGE_PATHS:
                gesture_image = load_and_resize_image(
                    IMAGE_PATHS[current_gesture], frame_height
                )
                if gesture_image is not None:
                    output_frame = np.concatenate((frame, gesture_image), axis=1)
                else:
                    output_frame = frame
                    cv2.putText(
                        output_frame,
                        "LOAD IMAGE FAILED - Check file names!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
            else:
                # Gestures without a paired image (FIST, OPEN_PALM, PEACE,
                # THUMBS_DOWN) — show only the camera feed with the colored label.
                output_frame = frame

            cv2.imshow('Gesture & Image Pairing', output_frame)

            key = cv2.waitKey(5)
            if key == ord('q') or key == 27:
                break
    finally:
        hand_landmarker.close()
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()