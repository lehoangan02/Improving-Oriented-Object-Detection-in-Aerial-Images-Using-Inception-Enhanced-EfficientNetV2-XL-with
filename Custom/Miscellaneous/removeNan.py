import os

label_folder = 'labelTxt'
image_folder = 'images'

# Iterate through all files in the label folder
for label_file in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_file)
    
    # Check if the label file is empty
    if os.path.getsize(label_path) == 0:
        # Construct the corresponding image file path
        image_file = label_file.replace('.txt', '.png')  # Assuming images are in .png format
        image_path = os.path.join(image_folder, image_file)
        
        # Remove the empty label file
        os.remove(label_path)
        print(f'Removed empty label file: {label_path}')
        
        # Remove the corresponding image file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f'Removed corresponding image file: {image_path}')