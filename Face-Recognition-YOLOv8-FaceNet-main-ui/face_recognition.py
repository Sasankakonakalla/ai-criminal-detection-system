import cv2
import pickle
import numpy as np
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# CONFIGURATION
# =====================================================

TARGET_IDENTITY = "Donald_Trump"
MATCH_THRESHOLD = 0.85

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "known_embeddings.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOG_FILE = os.path.join(LOG_DIR, "detections.txt")

FRAME_SKIP = 5

# ================= EMAIL CONFIG ======================
EMAIL_ENABLED = True

EMAIL_SENDER = os.getenv("ALERT_EMAIL")
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")   # Gmail App Password
EMAIL_RECEIVER = os.getenv("ALERT_EMAIL_RECEIVER")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_COOLDOWN_SECONDS = 60

# ================= INPUT MODE ========================
# Webcam can be triggered via:
# 1) use_webcam=True
# 2) Environment variable USE_WEBCAM=1
USE_WEBCAM = os.getenv("USE_WEBCAM") == "1"

# =====================================================
# INITIAL SETUP
# =====================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,identity,distance\n")

if not os.path.exists(EMBEDDINGS_FILE):
    print("[ERROR] known_embeddings.pkl not found")
    exit()

# =====================================================
# LOAD MODELS
# =====================================================

print("[INFO] Loading models...")
model = YOLO("detection/weights/best.pt")
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()
print("[INFO] Models loaded successfully.")

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

with open(EMBEDDINGS_FILE, "rb") as f:
    known_embeddings = pickle.load(f)

if TARGET_IDENTITY not in known_embeddings:
    print(f"[ERROR] {TARGET_IDENTITY} not in embeddings")
    exit()

trump_mean = np.mean(known_embeddings[TARGET_IDENTITY], axis=0)
print("[INFO] Known embeddings loaded.")

# =====================================================
# HELPERS
# =====================================================

last_email_time = {}

def log_detection(name, distance):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()},{name},{distance:.4f}\n")

def match_trump(embedding):
    dist = np.linalg.norm(embedding - trump_mean)
    if dist < MATCH_THRESHOLD:
        return TARGET_IDENTITY, dist
    return "Unknown", dist

def send_email_alert(name, image_path, distance):
    if not EMAIL_ENABLED:
        return

    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("[EMAIL WARNING] Email credentials not set")
        return

    now = datetime.now()
    if name in last_email_time:
        if (now - last_email_time[name]).total_seconds() < EMAIL_COOLDOWN_SECONDS:
            return

    last_email_time[name] = now

    try:
        msg = EmailMessage()
        msg["Subject"] = f"🚨 ALERT: {name} Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        msg.set_content(
            f"""
Suspicious Individual Detected

Identity : {name}
Distance : {distance:.4f}
Time     : {now.strftime('%Y-%m-%d %H:%M:%S')}

Evidence image attached.
"""
        )

        with open(image_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path)
            )

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL SENT] Alert sent for {name}")

    except Exception as e:
        print("[EMAIL ERROR]", e)

# =====================================================
# MAIN ANALYSIS (VIDEO + WEBCAM)
# =====================================================

def run_video_analysis(video_path=None, use_webcam=False):

    use_webcam = use_webcam or USE_WEBCAM

    if use_webcam:
        cap = cv2.VideoCapture(0)
        print("[INFO] Using WEBCAM input")
    else:
        video_path = video_path or os.getenv("VIDEO_PATH")
        if not video_path or not os.path.exists(video_path):
            print("[ERROR] Video file not found:", video_path)
            return
        cap = cv2.VideoCapture(video_path)
        print("[INFO] Using VIDEO:", video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)

        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = mtcnn(face_rgb)
            if face_tensor is None:
                continue

            if face_tensor.ndim == 4:
                face_tensor = face_tensor[0]

            embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().flatten()
            name, dist = match_trump(embedding)

            log_detection(name, dist)

            if name == TARGET_IDENTITY and frame_count % FRAME_SKIP == 0:
                person_dir = os.path.join(EVIDENCE_DIR, name)
                os.makedirs(person_dir, exist_ok=True)

                img_path = os.path.join(
                    person_dir,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )
                cv2.imwrite(img_path, face)
                send_email_alert(name, img_path, dist)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Trump Detection (Press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_video_analysis()