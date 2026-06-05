import time
import math
import threading
import winsound

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from ultralytics import YOLO

# ---- AYARLAR ----
EAR_THRESH = 0.21
DROWSY_SECONDS = 2.0
MAR_THRESH = 0.6
LOOK_AWAY_SECONDS = 3.0
HEAD_LEFT, HEAD_RIGHT = 0.35, 0.65
PHONE_CONF = 0.40
PHONE_EVERY = 5
MODEL_PATH = "face_landmarker.task"
EYE_MODEL_PATH = "eye_model.pt"
# -----------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = {"top": 13, "bottom": 14, "left": 61, "right": 291}
NOSE, FACE_LEFT, FACE_RIGHT = 1, 234, 454


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(p):
    h = dist(p[0], p[3])
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * h) if h > 0 else 0.0


class Alarm:
    def __init__(self):
        self.active = False
    def start(self):
        if not self.active:
            self.active = True
            threading.Thread(target=self._run, daemon=True).start()
    def stop(self):
        self.active = False
    def _run(self):
        while self.active:
            winsound.Beep(1200, 350)


# --- SENIN EGITTIGIN CNN MODELI ---
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


checkpoint = torch.load(EYE_MODEL_PATH, map_location='cpu', weights_only=False)
eye_cnn = EyeCNN()
eye_cnn.load_state_dict(checkpoint['model_state_dict'])
eye_cnn.eval()
IMG_SIZE = checkpoint.get('img_size', 64)
print("CNN modeli yuklendi: eye_model.pt")


def classify_eye(frame, lm, indices, h, w):
    xs = [int(lm[i].x * w) for i in indices]
    ys = [int(lm[i].y * h) for i in indices]
    pad = 15
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, w)
    y2 = min(max(ys) + pad, h)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return 1, 0.0
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    with torch.no_grad():
        out = eye_cnn(tensor)
        probs = torch.softmax(out, dim=1)
        pred = probs.argmax(1).item()
        conf = probs[0][pred].item()
    return pred, conf


alarm = Alarm()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

print("YOLO modeli yukleniyor...")
phone_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Kamera acilamadi.")
    raise SystemExit

timestamp_ms = 0
prev_time = 0.0
frame_no = 0
eye_closed_start = None
look_away_start = None
yawn_count = 0
mouth_was_open = False
phone_detected = False
phone_box = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_no += 1

    if frame_no % PHONE_EVERY == 0:
        yolo = phone_model(frame, classes=[67], conf=PHONE_CONF, imgsz=320, verbose=False)
        boxes = yolo[0].boxes
        if len(boxes) > 0:
            phone_detected = True
            phone_box = tuple(map(int, boxes[0].xyxy[0].tolist()))
        else:
            phone_detected = False
            phone_box = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    now = time.time()
    ear = mar = 0.0
    cnn_text = ""
    drowsy = distracted = False

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        def P(i):
            return (lm[i].x * w, lm[i].y * h)

        ear = (eye_aspect_ratio([P(i) for i in LEFT_EYE]) +
               eye_aspect_ratio([P(i) for i in RIGHT_EYE])) / 2.0
        mar = dist(P(MOUTH["top"]), P(MOUTH["bottom"])) / dist(P(MOUTH["left"]), P(MOUTH["right"]))
        ratio = (lm[NOSE].x - lm[FACE_LEFT].x) / (lm[FACE_RIGHT].x - lm[FACE_LEFT].x + 1e-6)

        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 180, 0), -1)

        # CNN ile goz siniflandirmasi
        l_pred, l_conf = classify_eye(frame, lm, LEFT_EYE, h, w)
        r_pred, r_conf = classify_eye(frame, lm, RIGHT_EYE, h, w)
        cnn_closed = (l_pred == 0 or r_pred == 0)
        cnn_conf = max(l_conf, r_conf)
        cnn_label = "Kapali" if cnn_closed else "Acik"
        cnn_text = f"CNN:{cnn_label} %{cnn_conf*100:.0f}"

        # Uyku tespiti: EAR VEYA CNN kapali derse -> goz kapali say
        eyes_closed = (ear < EAR_THRESH) or cnn_closed
        if eyes_closed:
            if eye_closed_start is None:
                eye_closed_start = now
            if now - eye_closed_start >= DROWSY_SECONDS:
                drowsy = True
        else:
            eye_closed_start = None

        if ratio < HEAD_LEFT or ratio > HEAD_RIGHT:
            if look_away_start is None:
                look_away_start = now
            if now - look_away_start >= LOOK_AWAY_SECONDS:
                distracted = True
        else:
            look_away_start = None

        if mar > MAR_THRESH:
            mouth_was_open = True
        elif mouth_was_open:
            yawn_count += 1
            mouth_was_open = False
    else:
        eye_closed_start = look_away_start = None

    if drowsy:
        status, color = "UYKULU! UYAN!", (0, 0, 255)
        alarm.start()
    elif phone_detected:
        status, color = "TELEFON! TEHLIKELI SURUS", (0, 0, 255)
        alarm.start()
    elif distracted:
        status, color = "DIKKAT DAGINIK! YOLA BAK!", (0, 0, 255)
        alarm.start()
    elif result.face_landmarks:
        status, color = "UYANIK", (0, 255, 0)
        alarm.stop()
    else:
        status, color = "YUZ YOK", (0, 0, 255)
        alarm.stop()

    if phone_detected and phone_box:
        x1, y1, x2, y2 = phone_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "TELEFON", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"DURUM: {status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR:{ear:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"MAR:{mar:.2f}", (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, cnn_text, (230, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Esneme:{yawn_count}", (410, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    n = time.time()
    fps = 1.0 / (n - prev_time) if prev_time else 0.0
    prev_time = n
    cv2.putText(frame, f"FPS:{int(fps)}", (w - 100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Surucu Takip (cikis: q)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

alarm.stop()
cap.release()
cv2.destroyAllWindows()
landmarker.close()