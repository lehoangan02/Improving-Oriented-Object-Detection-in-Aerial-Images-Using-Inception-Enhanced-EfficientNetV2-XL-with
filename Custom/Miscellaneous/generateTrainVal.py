import os

image_folder = 'images'
output_file = 'trainval.txt'

# Get all file names in the images folder
image_files = os.listdir(image_folder)

# Remove the .png extension from each file name
image_names = [os.path.splitext(file)[0] for file in image_files if file.endswith('.png')]

# Write the file names to trainval.txt
with open(output_file, 'w') as f:
    for name in image_names:
        f.write(name + '\n')

print(f'File names written to {output_file}')