import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def main():
    model_path = "pothole_detection/yolo11m_pothole_improved/weights/best.pt"
    
    video_files = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov")
    
    print(f"Found {len(video_files)} video files")
    
    os.makedirs("output_videos", exist_ok=True)
    
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    
    for video_file in video_files:
        print(f"\nProcessing video: {video_file}")
        output_dir = os.path.join("output_videos", os.path.splitext(video_file)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = os.path.join("output_videos", f"tracked_{os.path.basename(video_file)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            track_history = defaultdict(lambda: [])
            next_object_id = 0
            
            trackers = {}
            
            detection_interval = 1
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                original_frame = frame.copy()
                
                if frame_count % detection_interval == 0:
                    trackers = {}
                    
                    results = model.predict(frame, conf=0.25)[0]
                    
                    if len(results.boxes) > 0:
                        for box in results.boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = map(int, box[:4])
                            
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(original_frame, (x1, y1, x2-x1, y2-y1))
                            
                            trackers[next_object_id] = {
                                'tracker': tracker,
                                'class': results.names[int(results.boxes.cls[0])] if hasattr(results.boxes, 'cls') else "pothole",
                                'confidence': float(results.boxes.conf[0]) if hasattr(results.boxes, 'conf') else 1.0
                            }
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            track_history[next_object_id].append((center_x, center_y))
                            
                            next_object_id += 1
                
                to_delete = []
                for object_id, tracker_info in trackers.items():
                    success, box = tracker_info['tracker'].update(original_frame)
                    
                    if success:
                        x, y, w, h = map(int, box)
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        label = f"{tracker_info['class']} #{object_id}"
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        center_x = x + w // 2
                        center_y = y + h // 2
                        track_history[object_id].append((center_x, center_y))
                        
                        if len(track_history[object_id]) > 30:
                            track_history[object_id].pop(0)
                            
                        points = np.array(track_history[object_id], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                    else:
                        to_delete.append(object_id)
                
                for object_id in to_delete:
                    trackers.pop(object_id, None)
                
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                out.write(frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            print(f"Tracking completed. Output saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

if __name__ == "__main__":
    main()
    print("Detection and tracking completed using OpenCV")