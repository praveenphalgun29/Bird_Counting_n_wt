import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from collections import defaultdict
import pandas as pd

# Import our engine
from processor import process_video

app = FastAPI(title="Bird Counting & Weight API")

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "OK", "message": "Service is ready."}

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video, process it for bird counting/tracking, 
    and return JSON stats + link to annotated video.
    """
    try:
        temp_input_path = f"uploads/{file.filename}"
        output_filename = f"processed_{file.filename}"
        output_path = f"outputs/{output_filename}"

        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Starting processing for {file.filename}...")
        raw_data = process_video(temp_input_path, output_path)

        if raw_data is None:
            raise HTTPException(status_code=500, detail="Video processing failed.")

        df = pd.DataFrame(raw_data)
        
        response_data = {
            "filename": file.filename,
            "total_frames_processed": len(df['timestamp'].unique()) if not df.empty else 0,
            "artifacts": {
                "annotated_video": output_path
            }
        }

        if not df.empty:
            # Group by timestamp and count unique IDs
            counts_series = df.groupby('timestamp')['id'].nunique().to_dict()
            response_data["counts_timeseries"] = counts_series

            # Get the average weight index for each bird ID over its lifetime
            weight_stats = df.groupby('id')['weight_index'].mean().to_dict()
            response_data["weight_estimates"] = {
                str(bird_id): {"avg_weight_index": round(w, 2), "unit": "pixel_area_proxy"} 
                for bird_id, w in weight_stats.items()
            }
            
            response_data["max_birds_detected"] = int(df.groupby('timestamp')['id'].nunique().max())

        else:
            response_data["message"] = "No birds detected in the video."

        # Return the JSON
        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})