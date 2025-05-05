import os
import glob
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

def main():
    model_path = "pothole_detection/yolo11m_pothole_improved/weights/best.pt"
    
    video_files = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov")
    
    print(f"Found {len(video_files)} video files")
    
    os.makedirs("output_videos", exist_ok=True)
    
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    
    for video_file in video_files:
        print(f"\nProcessing video: {video_file}")
        output_path = os.path.join("output_videos", f"processed_{video_file}")
        
        try:
            model.track(
                source=video_file, 
                save=True, 
                conf=0.25, 
                project="output_videos", 
                name=os.path.splitext(video_file)[0],
                tracker="botsort.yaml"
            )
            print(f"Processed video with tracking saved to output_videos/{os.path.splitext(video_file)[0]}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

if __name__ == "__main__":
    main()
    print("Detection and tracking completed")