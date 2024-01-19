import glob
import os
import json

# Define the directory where the .txt files are located
input_directory = '../Outputs'
# Define the directory where the JSON files will be saved
output_directory = '../Finetune'

# Function to read the content of a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to write JSON data to a file
def write_json(file_name, data):
    with open(os.path.join(output_directory, file_name), 'w') as fout:
        json.dump(data, fout, indent=4)

# Function to parse the content of a file into the required JSON structure
def parse_content(file_content):
    # Split the content into lines and process each line as needed
    # Modify the parsing logic based on the structure of your .txt files
    lines = file_content.strip().split('\n')
    json_data = []
    prompt = ""

    image_file = ""
    output = ""
    for line in lines:
        # Extract the necessary information from each line
        
        if "Prompt" in line:
            prompt = line[8:]
            continue  # Skip to next line
        elif "Image path" in line:
            # Get image id from path (\img\cjvnzx1egq20g0804cxk22scu.jpeg)
            image_file = line[-30:]
            continue
        elif "[" in line:
            output = line

        if output == "":
            continue
        if image_file == "":
            continue
        if prompt == "":
            continue

        json_entry = {
            "id": image_file[:-5],
            "image": image_file,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": output
                }
            ]
        }

        json_data.append(json_entry)

        # Reset variables
        image_file = ""
        output = ""

    return json_data

# Find all .txt files in the input directory that contain "correct" in their name
for txt_file in glob.glob(os.path.join(input_directory, '*correct*.txt')):
    # Read the content of the file
    content = read_file(txt_file)
    # Parse the content into the required JSON format
    parsed_data = parse_content(content)
    # Generate a JSON file name based on the original .txt file name
    json_file_name = os.path.basename(txt_file).replace('.txt', '.json')
    # Write the JSON data to the output file
    write_json(json_file_name, parsed_data)