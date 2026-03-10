import os
import shutil
import random

def select_random_subset(dataset_path, output_path, percentage=10):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labelTxt")
    
    output_images_path = os.path.join(output_path, "images")
    output_labels_path = os.path.join(output_path, "labelTxt")
    
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_path) if f.endswith(".png")]
    sample_size = max(1, int(len(image_files) * (percentage / 100)))
    selected_files = random.sample(image_files, sample_size)
    
    for file in selected_files:
        image_src = os.path.join(images_path, file)
        label_src = os.path.join(labels_path, file.replace(".png", ".txt"))
        
        image_dst = os.path.join(output_images_path, file)
        label_dst = os.path.join(output_labels_path, os.path.basename(label_src))
        
        shutil.copy(image_src, image_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
    
    print(f"Copied {sample_size} images and labels to {output_path}")

if __name__ == "__main__":
    dataset_path = "./../../datasets/BridgeTrainFull"  # Change this to your dataset path
    output_path = "./../../datasets/GLH_10_Percent"  # Change this to your desired output folder
    select_random_subset(dataset_path, output_path)