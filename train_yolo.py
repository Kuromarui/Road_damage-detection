import os
import torch
import signal
import sys
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil
import subprocess

def signal_handler(sig, frame):
    print('\nTraining interrupted! Saving current state...')
    with open('stop', 'w') as f:
        f.write('Stop training')
    print('Created stop file. Training will stop after current batch.')
    sys.exit(0)

def train_yolo():
    
    signal.signal(signal.SIGINT, signal_handler)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_yaml = 'data_config.yaml'
    
    if os.path.exists('train/labels.cache'):
        os.remove('train/labels.cache')
    if os.path.exists('valid/labels.cache'):
        os.remove('valid/labels.cache')
        
    model = YOLO('yolo11m.pt')  

    try:
        results = model.train(
            data=data_yaml,    
            epochs=50,            
            imgsz=800,            
            batch=6,               
            patience=50,           
            device=device,             
            project='pothole_detection',  
            name='yolo11m_pothole_improved', 
            exist_ok=True,           
            verbose=True,
            cache='disk',
            workers=2,
            single_cls=False,
            rect=False,
            classes=None,
        )
        
        print("Exporting model to ONNX format...")
        model.export(format='onnx')
        
        print("Exporting model to TensorRT format...")
        try:
            model.export(format='engine', imgsz=800)
        except Exception as e:
            print(f"TensorRT export failed: {e}")
        
        print("Exporting model to OpenVINO format...")
        try:
            model.export(format='openvino', imgsz=800)
        except Exception as e:
            print(f"OpenVINO export failed: {e}")
            
    except Exception as e:
        print(f"Training error: {e}")
        raise
    finally:
        print("Training process completed.")

if __name__ == "__main__":
    train_yolo()
    print("Training completed successfully!") 