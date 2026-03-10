import streamlit as st
import os
import subprocess
import sys
import shutil
import time
import glob
import random
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.getcwd()
FACES_DB = os.path.join(PROJECT_ROOT, "faces_db")
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "detections.txt")
EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, "known_embeddings.pkl")

TARGET_IDENTITY = "Suspect"
PYTHON_EXEC = sys.executable

for d in [FACES_DB, VIDEOS_DIR, EVIDENCE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Surveillance",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state defaults ───────────────────────────────────────────────
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "embeddings_ready" not in st.session_state:
    st.session_state.embeddings_ready = os.path.exists(EMBEDDINGS_FILE)
if "augment_message" not in st.session_state:
    st.session_state.augment_message = None

# ── custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #6C63FF;
    --accent-light: #8B83FF;
    --success: #00D68F;
    --danger: #FF6B6B;
    --warning: #FFD93D;
    --card-bg: #161B22;
    --card-border: #30363D;
    --text-primary: #F0F6FC;
    --text-secondary: #8B949E;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── hero header ─────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 40%, #2D1B69 100%);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #00D68F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    position: relative;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.4);
    color: var(--accent-light);
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.8rem;
    position: relative;
}

/* ── metric cards ────────────────────────────────────── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0.3rem 0 0 0;
}
.metric-value.accent  { color: var(--accent); }
.metric-value.success { color: var(--success); }
.metric-value.danger  { color: var(--danger); }
.metric-value.warning { color: var(--warning); }

/* ── section cards ───────────────────────────────────── */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.section-icon.purple { background: rgba(108,99,255,0.2); }
.section-icon.green  { background: rgba(0,214,143,0.2); }
.section-icon.red    { background: rgba(255,107,107,0.2); }
.section-icon.yellow { background: rgba(255,217,61,0.2); }
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0;
}

/* ── step indicator ──────────────────────────────────── */
.step-bar {
    display: flex;
    gap: 0;
    margin-bottom: 1.5rem;
}
.step-item {
    flex: 1;
    text-align: center;
    padding: 0.7rem 0.5rem;
    position: relative;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    border-bottom: 3px solid var(--card-border);
    transition: all 0.2s;
}
.step-item.active {
    color: var(--accent-light);
    border-bottom-color: var(--accent);
}
.step-item.done {
    color: var(--success);
    border-bottom-color: var(--success);
}
.step-num {
    display: inline-block;
    width: 22px; height: 22px;
    line-height: 22px;
    border-radius: 50%;
    background: var(--card-border);
    color: var(--text-secondary);
    font-size: 0.7rem;
    margin-right: 0.4rem;
}
.step-item.active .step-num {
    background: var(--accent);
    color: #fff;
}
.step-item.done .step-num {
    background: var(--success);
    color: #fff;
}

/* ── image gallery grid ──────────────────────────────── */
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 0.6rem;
    margin-top: 0.8rem;
}
.gallery-thumb {
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid var(--card-border);
    aspect-ratio: 1;
    transition: border-color 0.2s, transform 0.2s;
}
.gallery-thumb:hover {
    border-color: var(--accent);
    transform: scale(1.04);
}
.gallery-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* ── sidebar styling ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
}
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.2rem 0;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 1.2rem;
}
.sidebar-brand-icon {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.sidebar-brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin: 0;
}

.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin: 1.2rem 0 0.5rem 0;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green  { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-dot.red    { background: var(--danger);  box-shadow: 0 0 6px var(--danger); }
.status-dot.yellow { background: var(--warning); box-shadow: 0 0 6px var(--warning); }

/* ── button overrides ────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s;
    border: 1px solid var(--card-border);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}

/* ── file uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 2px dashed var(--card-border);
    transition: border-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
}

/* ── tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--card-bg);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--card-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── toast-style messages ────────────────────────────── */
.stAlert {
    border-radius: 10px;
}

/* ── hide main menu, footer, and deploy (keep sidebar toggle only) ─ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Hide everything in header except first child (sidebar expand/collapse button) */
header > *:not(:first-child) { display: none !important; }
header a { display: none !important; }
/* Hide deploy/toolbar elements wherever they appear */
[data-testid="stToolbar"],
a[href*="share.streamlit"],
a[href*="streamlit.io"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────
def count_files(directory, extensions=("jpg", "jpeg", "png")):
    total = 0
    if os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            total += sum(1 for f in files if f.lower().rsplit(".", 1)[-1] in extensions)
    return total

def count_identities():
    if not os.path.isdir(FACES_DB):
        return 0
    return sum(1 for d in os.listdir(FACES_DB)
               if os.path.isdir(os.path.join(FACES_DB, d)))

def get_evidence_images(limit=20):
    images = []
    if os.path.isdir(EVIDENCE_DIR):
        for root, _, files in os.walk(EVIDENCE_DIR):
            for f in sorted(files, reverse=True):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(root, f))
                    if len(images) >= limit:
                        return images
    return images

def read_log_tail(n=20):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    return lines[-n:] if len(lines) > n else lines

# ── augmentation (from single photos to 50–60 variants each) ───────────
def _augment_one(img_bgr, seed):
    """Apply random augmentations to one image. img_bgr is numpy BGR (OpenCV)."""
    rng = random.Random(seed)
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # Horizontal flip (50% when chosen)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    # Small rotation (-15 to +15 degrees)
    if rng.random() < 0.7:
        angle = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Slight scale/crop (zoom 90–100%)
    if rng.random() < 0.5:
        scale = rng.uniform(0.92, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            out = cv2.resize(out, (new_w, new_h))
            pad_l = (w - new_w) // 2
            pad_t = (h - new_h) // 2
            padded = np.zeros_like(img_bgr)
            padded[pad_t : pad_t + new_h, pad_l : pad_l + new_w] = out
            out = padded

    # Brightness
    if rng.random() < 0.6:
        factor = rng.uniform(0.75, 1.25)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Contrast
    if rng.random() < 0.6:
        factor = rng.uniform(0.8, 1.2)
        mean = out.mean()
        out = np.clip((out.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    # Slight blur
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0.5)

    # Slight noise
    if rng.random() < 0.3:
        noise = np.random.randn(*out.shape).astype(np.float32) * rng.uniform(2, 8)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

def augment_identity(identity_name, per_image=55):
    """
    For each source image in faces_db/<identity_name>/ (no '_aug_' in name),
    generate per_image augmented versions and save as ..._aug_001.jpg etc.
    Returns (total_augmented_count, list of saved paths).
    """
    folder = os.path.join(FACES_DB, identity_name)
    if not os.path.isdir(folder):
        return 0, []

    exts = (".jpg", ".jpeg", ".png")
    source_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and "_aug_" not in f
    ]
    if not source_files:
        return 0, []

    saved_paths = []
    for src_name in source_files:
        src_path = os.path.join(folder, src_name)
        img = cv2.imread(src_path)
        if img is None:
            continue
        stem = os.path.splitext(src_name)[0]
        for i in range(per_image):
            aug = _augment_one(img, seed=hash((src_name, i)) % (2**32))
            out_name = f"{stem}_aug_{i:03d}.jpg"
            out_path = os.path.join(folder, out_name)
            cv2.imwrite(out_path, aug)
            saved_paths.append(out_path)

    return len(saved_paths), saved_paths

def get_augmented_images(identity_name, limit=60):
    """List paths of augmented images (_aug_ in filename) for an identity."""
    folder = os.path.join(FACES_DB, identity_name)
    if not os.path.isdir(folder):
        return []
    paths = []
    for f in sorted(os.listdir(folder)):
        if "_aug_" in f and f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(folder, f))
            if len(paths) >= limit:
                break
    return paths

# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🛡️</div>
        <p class="sidebar-brand-name">FaceGuard AI</p>
        <p class="sidebar-brand-sub">Surveillance Platform v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    # ── system status ─────────────────────
    st.markdown('<p class="sidebar-section-label">System Status</p>',
                unsafe_allow_html=True)

    embeddings_exist = os.path.exists(EMBEDDINGS_FILE)
    model_weights = os.path.exists(os.path.join(PROJECT_ROOT, "detection", "weights", "best.pt"))

    status_html = f"""
    <div style="display:flex; flex-direction:column; gap:6px; margin-bottom:0.8rem;">
        <span style="font-size:0.82rem; color:var(--text-primary);">
            <span class="status-dot {'green' if model_weights else 'red'}"></span>
            YOLOv8 Model
        </span>
        <span style="font-size:0.82rem; color:var(--text-primary);">
            <span class="status-dot {'green' if embeddings_exist else 'yellow'}"></span>
            Face Embeddings
        </span>
        <span style="font-size:0.82rem; color:var(--text-primary);">
            <span class="status-dot green"></span>
            FaceNet Engine
        </span>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)

    st.divider()

    # ── actions ───────────────────────────
    st.markdown('<p class="sidebar-section-label">Actions</p>',
                unsafe_allow_html=True)

    run_embeddings = st.button("Generate Embeddings", width="stretch")
    run_augment = st.button("Augment training images", width="stretch",
                            help="Create 55 variants per source photo (flip, rotate, brightness, etc.)")
    run_analysis = st.button("Start Analysis", width="stretch",
                             type="primary")

    st.divider()

    # ── settings ──────────────────────────
    st.markdown('<p class="sidebar-section-label">Configuration</p>',
                unsafe_allow_html=True)

    target_name = st.text_input("Target identity name", value=TARGET_IDENTITY,
                                help="Name for the person being tracked")
    match_threshold = st.slider("Match confidence threshold", 0.50, 1.00, 0.85, 0.01,
                                help="Lower = more matches (less strict)")
    use_webcam = st.toggle("Use webcam instead of video", value=False,
                           help="Enable live webcam feed for analysis")

    st.divider()

    # ── danger zone ───────────────────────
    st.markdown('<p class="sidebar-section-label">Danger Zone</p>',
                unsafe_allow_html=True)
    reset = st.button("Reset All Data", width="stretch")

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding-top:1rem; border-top:1px solid var(--card-border);">
        <p style="font-size:0.7rem; color:var(--text-secondary); margin:0;">
            Built with YOLOv8 + FaceNet + Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── hero banner ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">Face Recognition Surveillance</p>
    <p class="hero-subtitle">
        Real-time face detection and identification powered by YOLOv8 and FaceNet deep learning models
    </p>
    <span class="hero-badge">AI-Powered &bull; Real-Time &bull; Secure</span>
</div>
""", unsafe_allow_html=True)

# ── metrics row ──────────────────────────────────────────────────────────
training_count = count_files(FACES_DB)
evidence_count = count_files(EVIDENCE_DIR)
identity_count = count_identities()
video_count = len([f for f in os.listdir(VIDEOS_DIR)
                   if f.lower().endswith(".mp4")]) if os.path.isdir(VIDEOS_DIR) else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value accent">{training_count}</p>
        <p class="metric-label">Training Images</p>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value success">{identity_count}</p>
        <p class="metric-label">Identities Enrolled</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value warning">{evidence_count}</p>
        <p class="metric-label">Evidence Captured</p>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value danger">{video_count}</p>
        <p class="metric-label">Videos Processed</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── step progress indicator ──────────────────────────────────────────────
step1_done = training_count > 0
step2_done = os.path.exists(EMBEDDINGS_FILE)
step3_done = st.session_state.video_path is not None
step4_done = st.session_state.analysis_running

step_classes = [
    "done" if step1_done else "active",
    "done" if step2_done else ("active" if step1_done else ""),
    "done" if step3_done else ("active" if step2_done else ""),
    "done" if step4_done else ("active" if step3_done else ""),
]

st.markdown(f"""
<div class="step-bar">
    <div class="step-item {step_classes[0]}">
        <span class="step-num">1</span> Upload Faces
    </div>
    <div class="step-item {step_classes[1]}">
        <span class="step-num">2</span> Generate Embeddings
    </div>
    <div class="step-item {step_classes[2]}">
        <span class="step-num">3</span> Upload Video
    </div>
    <div class="step-item {step_classes[3]}">
        <span class="step-num">4</span> Run Analysis
    </div>
</div>
""", unsafe_allow_html=True)

# ── main content tabs ────────────────────────────────────────────────────
tab_upload, tab_evidence, tab_logs = st.tabs([
    "Upload & Analyze", "Evidence Gallery", "Detection Logs"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 – UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════
with tab_upload:
    # Show result of last "Augment training images" click (success or error)
    msg = st.session_state.get("augment_message")
    if msg is not None:
        kind, text = msg
        if kind == "success":
            st.success(text)
        else:
            st.error(text)
        st.session_state.augment_message = None

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── left: training images ──────────────────
    with col_left:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon purple">📸</div>
            <div>
                <p class="section-title">Training Images</p>
                <p class="section-desc">Upload clear face photos for identification</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        person_dir = os.path.join(FACES_DB, target_name)
        os.makedirs(person_dir, exist_ok=True)

        uploads = st.file_uploader(
            "Drag & drop face images here",
            accept_multiple_files=True,
            type=["jpg", "png", "jpeg"],
            key="face_uploader",
            help="Upload multiple clear front-facing photos of the target individual",
        )

        if uploads:
            for img in uploads:
                with open(os.path.join(person_dir, img.name), "wb") as f:
                    f.write(img.getbuffer())
            st.success(f"Successfully uploaded {len(uploads)} image(s) to **{target_name}**")

            st.markdown(f"**Preview** — {len(uploads)} uploaded image(s):")
            preview_cols = st.columns(min(len(uploads), 6))
            for idx, img in enumerate(uploads[:6]):
                with preview_cols[idx]:
                    try:
                        st.image(img, width="stretch")
                    except Exception:
                        st.caption("Preview N/A")
            if len(uploads) > 6:
                st.caption(f"... and {len(uploads) - 6} more")
        else:
            existing_count = count_files(os.path.join(FACES_DB, target_name))
            if existing_count > 0:
                st.info(f"**{existing_count}** training image(s) already stored for *{target_name}*. Upload more or proceed to generate embeddings.")
            else:
                st.info("No training images yet. Upload some above to get started.")

    # ── right: video upload ────────────────────
    with col_right:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon green">🎬</div>
            <div>
                <p class="section-title">Video Input</p>
                <p class="section-desc">Upload surveillance footage for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_video = st.file_uploader(
            "Drag & drop video file here",
            type=["mp4"],
            key="video_uploader",
            help="Upload an MP4 video file to scan for the target individual",
        )

        if uploaded_video:
            video_path = os.path.join(
                VIDEOS_DIR,
                f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.session_state.video_path = video_path
            st.success("Video uploaded successfully")
            st.video(video_path)
        elif st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.video(st.session_state.video_path)
        elif use_webcam:
            st.info("Webcam mode enabled — no video upload needed.")
        else:
            st.info("No video uploaded yet. Upload an MP4 above or enable webcam in settings.")

    # ── augmented training images gallery (so you can see the photos) ───
    aug_images = get_augmented_images(target_name, limit=120)
    person_dir = os.path.join(FACES_DB, target_name)
    source_count = len([
        f for f in (os.listdir(person_dir) if os.path.isdir(person_dir) else [])
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_aug_" not in f
    ]) if os.path.isdir(person_dir) else 0
    if aug_images:
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <div class="section-icon yellow">🔄</div>
            <div>
                <p class="section-title">Augmented training images</p>
                <p class="section-desc">Generated variants (flip, rotate, brightness, etc.) — use these for better recognition</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"**{len(aug_images)}** augmented image(s) for *{target_name}*. Showing preview below.")
        n_show = min(len(aug_images), 36)
        cols = 6
        for start in range(0, n_show, cols):
            row_cols = st.columns(cols)
            for c, idx in enumerate(range(start, min(start + cols, n_show))):
                with row_cols[c]:
                    try:
                        st.image(Image.open(aug_images[idx]), width="stretch")
                    except Exception:
                        st.caption("—")
        if len(aug_images) > n_show:
            st.caption(f"… and {len(aug_images) - n_show} more. Run **Generate Embeddings** to use all for training.")
    elif source_count > 0:
        st.markdown("---")
        st.info(f"You have **{source_count}** source photo(s) for *{target_name}*. Click **\"Augment training images\"** in the sidebar (under Actions) to generate ~55 variants per photo. They will appear here.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 – EVIDENCE GALLERY
# ═══════════════════════════════════════════════════════════════════════
with tab_evidence:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon red">🔍</div>
        <div>
            <p class="section-title">Evidence Gallery</p>
            <p class="section-desc">Captured face crops from detected matches</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    evidence_images = get_evidence_images(limit=30)

    if evidence_images:
        ev_cols = st.columns(6)
        for idx, img_path in enumerate(evidence_images):
            with ev_cols[idx % 6]:
                try:
                    st.image(Image.open(img_path), width="stretch")
                    fname = os.path.basename(img_path)
                    parts = fname.replace(".jpg", "").split("_")
                    if len(parts) >= 2:
                        st.caption(f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}")
                except Exception:
                    st.caption("Error loading")
        st.caption(f"Showing {len(evidence_images)} most recent captures")
    else:
        st.info("No evidence captured yet. Run an analysis to start collecting detections.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 – DETECTION LOGS
# ═══════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon yellow">📋</div>
        <div>
            <p class="section-title">Detection Logs</p>
            <p class="section-desc">Recent detection events with timestamps and distances</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    log_lines = read_log_tail(50)

    if len(log_lines) > 1:
        import pandas as pd

        rows = []
        for line in log_lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                rows.append({
                    "Timestamp": parts[0],
                    "Identity": parts[1],
                    "Distance": parts[2],
                })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Showing last {len(rows)} detection event(s)")
        else:
            st.info("Log file exists but has no parseable entries yet.")
    else:
        st.info("No detection logs available. Logs are created when you run an analysis.")

# ── handle actions ───────────────────────────────────────────────────────
if reset:
    shutil.rmtree(FACES_DB, ignore_errors=True)
    shutil.rmtree(VIDEOS_DIR, ignore_errors=True)
    shutil.rmtree(EVIDENCE_DIR, ignore_errors=True)
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    for d in [FACES_DB, VIDEOS_DIR, EVIDENCE_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    st.session_state.video_path = None
    st.session_state.analysis_running = False
    st.session_state.embeddings_ready = False
    st.toast("All data has been reset", icon="🗑️")
    time.sleep(0.5)
    st.rerun()

if run_embeddings:
    if count_files(FACES_DB) == 0:
        st.toast("Upload training images first", icon="⚠️")
    else:
        with st.spinner("Generating face embeddings — this may take a moment..."):
            subprocess.run([PYTHON_EXEC, "generate_face_embeddings.py"])
        st.session_state.embeddings_ready = True
        st.toast("Embeddings generated successfully!", icon="✅")
        st.rerun()

if run_augment:
    st.session_state.augment_message = None
    person_dir = os.path.join(FACES_DB, target_name)
    if not os.path.isdir(person_dir):
        st.session_state.augment_message = ("error", f"No folder for identity **{target_name}**. Upload at least one training image first.")
    else:
        source_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_aug_" not in f
        ]
        if not source_files:
            st.session_state.augment_message = ("error", f"No source photos found for **{target_name}** (only images without '_aug_' in the name are used). Upload at least one training image.")
        else:
            try:
                with st.spinner("Creating augmented images (55 per source photo)…"):
                    n, paths = augment_identity(target_name, per_image=55)
                st.session_state.augment_message = ("success", f"Created **{n}** augmented images from **{len(source_files)}** source photo(s). See the gallery below.")
            except Exception as e:
                st.session_state.augment_message = ("error", f"Augmentation failed: {e}")
    st.rerun()

if run_analysis:
    has_video = st.session_state.video_path and os.path.exists(st.session_state.video_path)

    if not has_video and not use_webcam:
        st.toast("Upload a video or enable webcam first", icon="⚠️")
    elif not os.path.exists(EMBEDDINGS_FILE):
        st.toast("Generate embeddings before running analysis", icon="⚠️")
    else:
        env = os.environ.copy()
        env["MATCH_THRESHOLD"] = str(match_threshold)

        if use_webcam:
            env["USE_WEBCAM"] = "1"
        elif st.session_state.video_path:
            env["VIDEO_PATH"] = st.session_state.video_path

        subprocess.Popen([PYTHON_EXEC, "face_recognition.py"], env=env)
        st.session_state.analysis_running = True
        st.toast("Analysis started! Press **Q** in the video window to stop.", icon="🚀")
