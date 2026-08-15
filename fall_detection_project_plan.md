# Elderly Fall Detection System — Final Project Plan

## Overview

A video-based fall detection system that processes uploaded video files and detects fall events using skeleton tracking and temporal action recognition. No camera required — upload a video, get back an annotated video with fall events highlighted + a JSON event log.

**Input:** Video file (.mp4, .avi, .mov)
**Output:** Annotated video (skeletons + fall alerts drawn on frames) + JSON log of all detected events

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.10+ | All libraries are Python-first |
| Video I/O | OpenCV | Read/write video files, draw overlays |
| Detection + Pose | YOLO26-pose (Ultralytics) | Person bbox + 17 keypoints in one pass |
| Tracking | BoT-SORT (built into Ultralytics) | Re-ID after occlusion, camera motion compensation |
| Temporal classifier | LSTM (PyTorch) | Classify 30-frame skeleton sequences |
| Web UI (optional) | Streamlit or Gradio | Upload video → view results in browser |
| Backend (optional) | FastAPI | REST API for integration |

---

## Phase 1 — Video input + detection + tracking (Week 1)

**Goal:** Read a video file, detect persons with skeletons, track them with BoT-SORT.

### Steps

1. Set up environment:
```bash
pip install ultralytics opencv-python torch torchvision numpy
```

2. Write the core processing script:
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")

# Process a video file (not camera)
results = model.track(
    source="test_video.mp4",
    tracker="botsort.yaml",
    stream=True,
    persist=True
)

for r in results:
    frame = r.plot()  # draws boxes + skeletons + track IDs
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

3. Save the annotated output as a new video file using OpenCV VideoWriter
4. Test with a sample video — verify skeletons and track IDs are drawn correctly
5. Test occlusion: person walks behind furniture and comes back — confirm BoT-SORT keeps the same ID

### Deliverable
Script that reads a video file → outputs annotated video with tracked skeletons.

---

## Phase 2 — Feature extraction from skeletons (Week 2)

**Goal:** Extract fall-relevant features from skeleton keypoints for each tracked person per frame.

### Features to extract (per frame, per person)

| Feature | How to compute | Why it matters |
|---|---|---|
| Body angle | Angle between hip-midpoint→nose line and vertical | Standing ≈ 0°, lying ≈ 90° |
| Center of gravity height | Average y-coordinate of hips, normalized by frame height | Drops suddenly during a fall |
| Vertical velocity | Change in CoG height between consecutive frames | Sudden downward spike = fall |
| Bbox aspect ratio | Bounding box width / height | Standing = tall/narrow, fallen = wide/short |
| Keypoint confidence | Average confidence of 17 keypoints | Low = person partially occluded |

### Steps

5. Write `extract_features(keypoints, bbox)` function
6. Store features in a dictionary keyed by BoT-SORT track ID:
```python
from collections import deque
tracks = {}  # {track_id: deque(maxlen=30)}
```
7. For each frame, for each detected person:
   - Get track ID from BoT-SORT
   - Compute 5 features from skeleton
   - Append to that track's deque
8. Test: print features while processing a video, verify they change during falls

### Deliverable
Feature extraction module producing per-track feature sequences in real-time.

---

## Phase 3 — Dataset preparation (Week 2-3)

**Goal:** Convert public fall video datasets into labeled training data for the LSTM.

### Datasets to download

| Dataset | Source | Content |
|---|---|---|
| UR Fall Detection | http://fenix.ur.edu.pl/~mkepski/ds/uf.html | 70 videos (30 falls, 40 normal activities) |
| Le2i Fall Dataset | Available on Kaggle | Videos from 4 rooms with frame-level annotations |
| MCFD | Publicly available | Multi-camera fall scenarios |

### Processing pipeline

```
Downloaded videos + labels
    ↓  Run YOLO26-pose + BoT-SORT on every frame
Skeleton keypoints per person per frame
    ↓  Compute body angle, velocity, bbox ratio, etc.
Feature vectors (5 values per frame)
    ↓  Slice into 30-frame sliding windows
    ↓  Label each window: 0=normal, 1=fall, 2=on ground
Training samples saved as .npy files
```

### Steps

9. Download UR Fall Detection + Le2i datasets
10. Write a batch processing script:
```python
for video in dataset_videos:
    results = model.track(source=video, tracker="botsort.yaml", stream=True, persist=True)
    for r in results:
        # extract features per person per frame
        # store in sequences keyed by track_id
    # slice into 30-frame windows
    # assign labels based on dataset annotations
```
11. Save as `X_train.npy` (shape: N × 30 × 5) and `y_train.npy` (shape: N)
12. Split: 80% train / 10% validation / 10% test

### Deliverable
Clean `.npy` files ready for LSTM training. Approximately 2,000-5,000 training samples.

---

## Phase 4 — Train the LSTM classifier (Week 3-4)

**Goal:** A model that classifies 30-frame skeleton sequences into normal / fall / on-ground.

### Model architecture

```python
import torch.nn as nn

class FallDetector(nn.Module):
    def __init__(self, input_size=5, hidden=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):  # x shape: (batch, 30, 5)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # classify using last timestep
```

### Steps

13. Create PyTorch Dataset and DataLoader for the `.npy` files
14. Train with:
    - Loss: CrossEntropyLoss (with class weights if imbalanced)
    - Optimizer: Adam, lr=0.001
    - Epochs: 50-100
    - Early stopping on validation loss
15. Target metrics:
    - Overall accuracy: > 95%
    - Fall recall: > 90% (missing a fall is dangerous)
    - False positive rate: < 5%
16. Save best model checkpoint

### Optional experiment
Try replacing LSTM with a small Transformer encoder to compare performance.

### Deliverable
Trained `fall_detector.pth` model checkpoint + training metrics report.

---

## Phase 5 — End-to-end pipeline (Week 5)

**Goal:** Upload a video → get annotated output video + event log.

### Pipeline flow

```
input_video.mp4
    → OpenCV reads frames
    → YOLO26-pose detects persons + keypoints
    → BoT-SORT assigns persistent track IDs
    → Feature extraction per person per frame
    → Sliding window buffer (30 frames per track)
    → LSTM classifies each window
    → If confidence > 0.85 → mark as fall event
    → Cooldown: suppress duplicate alerts for same track for 60s
    → Draw alert overlay on frame + write to output video
    → Log event to JSON
output_annotated.mp4 + events.json
```

### Steps

17. Integrate all modules into a single `process_video(input_path, output_path)` function
18. Add visual overlays on fall detection:
    - Red bounding box around the person
    - "FALL DETECTED" text on frame
    - Skeleton drawn in red instead of green
    - Timestamp overlay
19. Generate JSON event log:
```json
{
  "events": [
    {
      "type": "fall",
      "track_id": 1,
      "start_frame": 342,
      "end_frame": 372,
      "timestamp": "00:00:11.4",
      "confidence": 0.93
    }
  ],
  "summary": {
    "total_falls": 1,
    "persons_tracked": 2,
    "video_duration": "00:01:30"
  }
}
```
20. Test with:
    - Fall videos from the test split of your dataset
    - YouTube fall detection test videos (plenty available)
    - Tricky false positives: sitting down quickly, picking up objects, yoga poses, lying on couch

### Deliverable
Complete CLI tool:
```bash
python detect_falls.py --input video.mp4 --output result.mp4
```

---

## Phase 6 — Web UI (Week 6)

**Goal:** A simple web interface for non-technical users to upload and view results.

### Option A: Streamlit (simplest)

```python
import streamlit as st

st.title("Fall Detection System")
uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded:
    # save to temp file
    # process with pipeline
    # show annotated video
    st.video(output_path)
    st.json(event_log)
    st.download_button("Download annotated video", output_data)
```

### Option B: Gradio (good for ML demos / Hugging Face deployment)

```python
import gradio as gr

def process(video_path):
    output_path, events = detect_falls(video_path)
    return output_path, events

demo = gr.Interface(
    fn=process,
    inputs=gr.Video(),
    outputs=[gr.Video(), gr.JSON()]
)
demo.launch()
```

### Steps

21. Build the UI with Streamlit or Gradio
22. Add a progress bar during processing
23. Display the annotated video with fall events highlighted
24. Show the JSON event log with timestamps
25. Add a download button for the output video

### Deliverable
Web app where you upload video → see results in browser → download output.

---

## Phase 7 (Bonus) — Production touches

### 7a. Hugging Face Spaces deployment
- Deploy Gradio app to Hugging Face Spaces for free hosting
- Anyone can try your demo without installing anything
- Great for portfolio — just share the link

### 7b. FastAPI backend
- REST API: `POST /detect` with video file → returns annotated video + JSON
- Swagger docs auto-generated
- Can integrate with any frontend or mobile app

### 7c. Edge deployment (if you get hardware later)
- Export YOLO26-pose to ONNX → TensorRT
- Export LSTM to ONNX
- Run on Jetson Nano / Orin for real-time camera feed
- Benchmark FPS and latency

---

## Project Structure

```
fall-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── botsort.yaml              # BoT-SORT tracker config
├── data/
│   ├── raw/                      # downloaded dataset videos
│   ├── processed/                # extracted .npy features
│   └── splits/                   # train/val/test splits
├── models/
│   ├── fall_detector.py          # LSTM model definition
│   └── checkpoints/              # saved .pth files
├── src/
│   ├── feature_extraction.py     # skeleton → feature vectors
│   ├── dataset_processor.py      # batch process datasets → .npy
│   ├── train.py                  # LSTM training loop
│   ├── pipeline.py               # full end-to-end inference
│   └── utils.py                  # drawing, logging helpers
├── app/
│   ├── streamlit_app.py          # web UI
│   └── gradio_app.py             # alternative web UI
├── tests/
│   └── test_pipeline.py          # unit tests
└── notebooks/
    ├── 01_exploration.ipynb       # dataset exploration
    ├── 02_training.ipynb          # model training experiments
    └── 03_evaluation.ipynb        # metrics and analysis
```

---

## Timeline Summary

| Phase | What | Duration |
|---|---|---|
| Phase 1 | Video input + YOLO26-pose + BoT-SORT | Week 1 |
| Phase 2 | Feature extraction from skeletons | Week 2 |
| Phase 3 | Dataset download + processing | Week 2-3 |
| Phase 4 | LSTM training | Week 3-4 |
| Phase 5 | End-to-end pipeline + CLI tool | Week 5 |
| Phase 6 | Web UI (Streamlit/Gradio) | Week 6 |
| Phase 7 | Bonus: HuggingFace / FastAPI / Edge | Week 7-8 |

**Total: 6 weeks core + 2 weeks optional extras**

---

## What This Demonstrates on Your Portfolio

- Real-time video processing (OpenCV)
- State-of-the-art object detection (YOLO26)
- Human pose estimation (YOLO26-pose)
- Multi-object tracking (BoT-SORT)
- Temporal sequence modeling (LSTM / PyTorch)
- Full ML pipeline (data processing → training → inference → deployment)
- Web application (Streamlit / Gradio)
- Production engineering (JSON logging, confidence filtering, cooldown logic)
