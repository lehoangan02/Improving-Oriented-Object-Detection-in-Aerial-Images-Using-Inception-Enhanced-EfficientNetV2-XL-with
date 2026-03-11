import os

def combine_txt_files(input_folder, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(input_folder, filename), 'r') as infile:
                    outfile.write(infile.read())
                    # outfile.write("\n")  # Add a newline after each file's content

input_folder = './labelTxt'  # Replace with your label folder path
output_file = 'annoBridge.txt'  # Replace with your desired output file name

combine_txt_files(input_folder, output_file)