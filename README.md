# AI-Powered Criminal Detection System

Real-time surveillance and face recognition for criminal identification using **YOLOv8** (face detection) and **FaceNet** (face embeddings and matching).

## Features

- **Real-time face detection** — YOLOv8 for fast, accurate face detection in images and video
- **Face recognition** — FaceNet embeddings to match faces against a known database
- **Streamlit web UI** — Upload videos, manage identities, generate embeddings, and run analysis from the browser
- **Evidence & logging** — Save detection snapshots and logs for review

## Project Structure

```
├── Face-Recognition-YOLOv8-FaceNet-main-ui/
│   ├── app.py                    # Streamlit surveillance UI
│   ├── face_recognition.py       # Recognition pipeline (YOLOv8 + FaceNet)
│   ├── generate_face_embeddings.py
│   ├── detection/
│   │   ├── yolov8_detector.py
│   │   └── yolov8_trainer.py
│   ├── faces_db/                 # Training images per identity
│   ├── videos/                   # Uploaded videos
│   ├── evidence/                 # Detection snapshots
│   └── logs/                     # Detection logs
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sasankakonakalla/ai-criminal-detection-system-.git
   cd ai-criminal-detection-system-
   ```

2. **Go to the app directory and create a virtual environment**
   ```bash
   cd Face-Recognition-YOLOv8-FaceNet-main-ui
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **In the UI**
   - Add training images for each identity (e.g. suspect) under **faces_db** (or via the app).
   - Run **Generate Embeddings** to build the face database.
   - Upload a video and run **Analysis** to detect and match faces in real time.

3. **Optional: CLI**
   - Generate embeddings: `python generate_face_embeddings.py`
   - Run recognition: `python face_recognition.py` (adjust paths/identity in the script as needed)

## Requirements

- Python 3.8+
- See `Face-Recognition-YOLOv8-FaceNet-main-ui/requirements.txt` for packages (e.g. Streamlit, OpenCV, PyTorch, facenet-pytorch, ultralytics).

## License

This project is for educational and research purposes. Use responsibly and in compliance with local laws and privacy regulations.
