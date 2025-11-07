# Multimodal Biometric Recognition System (Face + Ear)

This project implements a **real-time multimodal biometric authentication system** that combines **face** and **ear** recognition for enhanced identification accuracy.  
It leverages **ArcFace (InsightFace)** for face recognition, **YOLOv8** for ear detection, and **ResNet50** for ear feature extraction.

---

## Features

- Face Detection and Recognition using ArcFace (InsightFace)
- Ear Detection using YOLOv8 (custom-trained model)
- Ear Feature Extraction using ResNet50
- Multimodal Fusion (Face + Ear) for high reliability
- Guided Registration with multi-angle capturing (frontal, left, right, up, down)
- Real-Time Recognition using a live webcam feed
- Automatic Database Management (embeddings stored and retrieved automatically)
- Fallback to single-modality recognition (face-only or ear-only)
- Configurable Confidence Thresholds

---

## Tech Stack

| Component | Model / Library | Purpose |
|------------|----------------|----------|
| Face Recognition | ArcFace (InsightFace) | Extracts face embeddings |
| Ear Detection | YOLOv8 | Detects human ears |
| Ear Encoding | ResNet50 | Generates ear feature vectors |
| Similarity Metric | Cosine Similarity | Measures embedding similarity |
| Database | Pickle (.pkl) | Stores biometric data |
| Frameworks | OpenCV, PyTorch, Scikit-learn | Core ML and vision pipeline |

---

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/multimodal-biometric-system.git
cd multimodal-biometric-system
```
### Step 2: Install Dependencies
```bash
pip install -r requirementscctv.txt
```

### Step 3: Add your trained YOLO ear model
Place your trained YOLOv8 ear detection model file (for example, best_3.pt) in the project root directory.

---

## How It Works

### Initialization

- Loads ArcFace (`buffalo_l`) model via InsightFace for face recognition.
- Loads YOLOv8 model for ear detection.
- Initializes ResNet50 for ear feature extraction.
- Loads all registered users' embeddings from the `biometric_database` directory.

### Registration

- Captures multiple face and ear samples across different head angles (`frontal`, `left`, `right`, `up`, `down`).
- Extracts embeddings for both face and ear.
- Saves averaged embeddings and metadata in `.pkl` format under the `biometric_database/` directory.
- Automatically reloads the updated database after registration.

### Recognition

- Captures live frames from the webcam.
- Detects faces and ears.
- Extracts embeddings from both modalities.
- Compares embeddings with the stored database using cosine similarity.
- Fuses confidence scores when both modalities are present.
- Displays bounding boxes and labels (name + confidence score) on the video feed.

---

## Recognition Logic

### Fusion Formula
Combined Confidence = (Face Similarity + Ear Similarity) / 2


---

### Matching Scenarios

| Case | Modalities Detected | Decision |
|------|----------------------|-----------|
| Both | Face + Ear | Fuse both embeddings for final decision |
| Face Only | Face | Match using face embeddings |
| Ear Only | Ear | Match using ear embeddings |
| None | None | Unknown person |

---

## Accuracy Estimation

| System | Typical Accuracy |
|---------|------------------|
| Face Recognition (ArcFace) | 97% – 99% |
| Ear Recognition (YOLO + ResNet50) | 85% – 92% |
| Combined (Face + Ear) | 98.5% – 99.7% |

---

### Theoretical Justification

When combining two independent modalities:
P(error_combined) = P(face_error) × P(ear_error)

**Example:**

If `P(face_error) = 3%` and `P(ear_error) = 8%`, then:
P(error_combined) = 0.03 × 0.08 = 0.0024
Accuracy = 99.76%


---

## Factors Affecting Accuracy

| Factor | Impact |
|--------|--------|
| Lighting | Poor lighting reduces both face and ear recognition quality |
| Pose Angle | Large head rotations (>45°) decrease detection accuracy |
| Occlusion | Hair or accessories can block ear or face visibility |
| Camera Quality | Low-resolution cameras degrade embedding quality |
| Ear Detection Model | Weak YOLO models reduce ear recognition reliability |

---

## Thresholds Used

| Parameter | Default Value | Description |
|------------|---------------|-------------|
| FACE_THRESHOLD | 0.35 | Minimum cosine similarity for face recognition |
| EAR_THRESHOLD | 0.50 | Minimum cosine similarity for ear recognition |
| COMBINED_CONFIDENCE | 0.98 | Minimum combined confidence for final match |

---

## Folder Structure
multimodal-biometric-system/
│
├── biometric_database/ # Stores .pkl files of registered users
├── best_3.pt # YOLOv8 ear detection model
├── main.py # Main program file
├── requirements.txt
└── README.md


## Usage

### Run the Program

```bash
python main.py
```

#### ✍️ Author
Developed by **Aman Kumar Dwiwedi**
