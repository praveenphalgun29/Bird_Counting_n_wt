# Bird Counting & Weight Estimation

## Overview
This project implements a computer vision pipeline to monitor poultry in a farm setting using CCTV footage. It provides an API to process video files and return:
1. **Bird Counts Over Time:** Using YOLOv8 detection and BoT-SORT/ByteTrack for stable object tracking.
2. **Weight Estimation Proxy:** Using Instance Segmentation masks to calculate pixel area as a proxy for bird weight.

## Approach & Methodology

### 1. Detection & Tracking
* **Model:** YOLOv8 Nano Segmentation (`yolov8n-seg`).
* **Tracking:** We utilize the tracking capabilities inherent in Ultralytics (BoT-SORT/ByteTrack) to assign unique IDs to birds. This ensures we count unique instances rather than raw detections per frame.
* **Logic:** The system tracks `class: 14` (bird) from the COCO dataset. (Note: For production, a custom-trained model on specific poultry breeds would improve accuracy).

### 2. Weight Estimation (Proxy)
Since ground-truth weight data (grams) was not available for the provided video, we implemented a **Pixel-Area Proxy**:
* **Method:** We extract the segmentation mask for each tracked bird.
* **Calculation:** `Weight_Index = Pixel_Area / 100`
* **Conversion to Grams:** To convert this index to real weight, the system would require:
    1.  **Camera Calibration:** To map pixel dimensions to real-world units ($cm^2$).
    2.  **Regression Model:** A simple linear regression mapping the surface area ($cm^2$) to mass ($g$) based on the specific bird breed.

## Setup & Installation

### Prerequisites
* Python 3.8+
* Virtual Environment (Recommended)

### Installation
1.  Clone this repository
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `ultralytics`, `fastapi`, `uvicorn`, `opencv-python` are installed)*

## Running the API

1.  Start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```
2.  The API will be live at `http://127.0.0.1:8000`.

## API Usage

### Endpoint: `POST /analyze_video`
Uploads a video file for processing.

**Example using Curl:**
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/analyze_video](http://127.0.0.1:8000/analyze_video)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_chicken.mp4;type=video/mp4'

```

**Response Format:**

```json
{
  "filename": "video.mp4",
  "total_frames_processed": 532,
  "counts_timeseries": {
    "0.04": 4,
    "1.0": 5
  },
  "weight_estimates": {
    "1": {"avg_weight_index": 426.25, "unit": "pixel_area_proxy"}
  }
}

```

