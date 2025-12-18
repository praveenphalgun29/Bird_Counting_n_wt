import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def process_video(input_path, output_path):
    """
    Processes a video to detect, track, and estimate weight of birds.
    """
    # 1. Load the YOLOv8 Segmentation Model
    model = YOLO('yolov8n-seg.pt')

    # 2. Open the Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return None

    # Get video properties for saving the output
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Setup Video Writer (to save the annotated video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bird_data = [] # List to store data for JSON response

    print("Processing video... (Press 'q' to stop early)")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        timestamp = frame_count / fps

        # 4. Run Tracking
        results = model.track(frame, persist=True, verbose=False, classes=[14])

        if results[0].boxes.id is not None:
            # Get the boxes, IDs, and masks
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            # Check if masks are available (segmentation)
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy() 
                
                
                current_count = len(track_ids)

                # Loop through each detected bird
                for box, track_id, mask in zip(boxes, track_ids, results[0].masks):
                    # --- WEIGHT ESTIMATION LOGIC ---
                    # Calculate the area (number of pixels) of the bird mask
                    poly = mask.xy[0] 
                    
                    # Calculate area using OpenCV
                    pixel_area = cv2.contourArea(poly.astype(np.float32))
                    weight_index = round(pixel_area / 100, 2)

                    # --- ANNOTATION ---
                    x1, y1, x2, y2 = box
                    
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"ID:{track_id} W:{weight_index}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.polylines(frame, [poly.astype(int)], True, (0, 0, 255), 2)

                    # Save Data for JSON
                    bird_data.append({
                        "timestamp": round(timestamp, 2),
                        "id": int(track_id),
                        "weight_index": weight_index
                    })

                # Draw Total Count on Screen
                cv2.putText(frame, f"Count: {current_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Write frame to output video
        out.write(frame)
        
        cv2.imshow("Bird Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing Complete. Saved to {output_path}")
    return bird_data

if __name__ == "__main__":
    process_video("sample_chicken.mp4", "output_test.mp4")