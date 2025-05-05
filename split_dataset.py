import os
import shutil
import random
from pathlib import Path

def split_dataset(split_ratio=0.2):

    train_img_dir = os.path.join('images', 'train')
    train_label_dir = os.path.join('labels', 'train')
    val_img_dir = os.path.join('images', 'val')
    val_label_dir = os.path.join('labels', 'val')
    
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    train_images = list(Path(train_img_dir).glob('*'))
    num_images = len(train_images)
    num_val = int(num_images * split_ratio)
    
    val_images = random.sample(train_images, num_val)
    
    print(f"Total images: {num_images}")
    print(f"Moving {num_val} images to validation set")
    

    for img_path in val_images:

        label_filename = img_path.stem + '.txt'
        label_path = Path(os.path.join(train_label_dir, label_filename))
        
        if not label_path.exists():
            print(f"Warning: No label file found for {img_path.name}")
            continue
        
        val_img_dest = Path(os.path.join(val_img_dir, img_path.name))
        val_label_dest = Path(os.path.join(val_label_dir, label_filename))
        
        shutil.move(str(img_path), str(val_img_dest))
        shutil.move(str(label_path), str(val_label_dest))
        
        print(f"Moved {img_path.name} and its label to validation set")

if __name__ == "__main__":
    split_dataset()
    print("Dataset split complete!") 