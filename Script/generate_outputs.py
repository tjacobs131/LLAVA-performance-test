import replicate
import os
import glob
import json

use_api = False # Determines if the API will be called or not

# --- Setup ----

def find_in_dict(key, dictionary):
    if isinstance(dictionary, dict):
        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for result in find_in_dict(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_in_dict(key, d):
                        yield result
    elif isinstance(dictionary, list):
        for d in dictionary:
            for result in find_in_dict(key, d):
                yield result

# Get api key from file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.abspath(os.path.join(script_dir, "..", "..", "key.txt")) # Path to the API key file (located in the directory outside of the repository)
with open(key_file_path, "r") as key_file:
    api_key = key_file.read()

os.environ["REPLICATE_API_TOKEN"] = str(api_key)

outputs = []
image_count = 0

if use_api:

    # --- Prepare data ---
    
    prompts = ["You are an expert in plant disease detection on leaves, are there any diseases that can you detect in this image?, your options are: rust (mild)., rust (extreme)., healthy.. You are a part of an automatic farming control loop, therefore a final answer is absolutely necessary. Your job is to reason about the health status about the leaf that is provided. Output your final answer in these brackets: []. It does not matter how sure you are about your answer, it is crucial that you make your best guess anyway. An example of a good output: Based on the image, the leaf appears to have yellow spots in a localised area. The presence of yellow spots on the leaf indicates that the plant is experiencing stress or damage due to rust, the affected area seems small compared to the size of the leaf. Therefore, the final answer is: [rust (mild)]"]

    print("Amount of prompts: " + str(len(prompts)))

    # Get images
    image_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "img"))
    image_paths = glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)

    print("Amount of requests: " + str(len(image_dir) * len(prompts)))
    print("Estimated run time: " + str(len(image_dir) * len(prompts) * 5) + " sec.")

    # --- Get outputs from the LVLM ---

    for prompt in prompts:
        print("--- Prompt: " + prompt + " ---")

        for image_path in image_paths:
            print("\n\nImage #" + str(image_count) +": "  + image_path)
            outputs.append("")

            with open(image_path, "rb") as image_file:
                output = replicate.run(
                    "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
                    input={"prompt": prompt,
                    "image": image_file}
                )

                for item in output:
                    outputs[image_count] += item
                    print(item, end="")

                image_count += 1    

        print("\n")

else: # Not using the API

    # --- Get outputs from file ---

    outputs_path = os.path.abspath(os.path.join(script_dir, "..", "Outputs", "example_outputs.txt"))
    with open (outputs_path, "r") as outputs_file:
        outputs_text = outputs_file.read()
        outputs = outputs_text.split("\n\n")

# --- Analyse outputs ---

rust_levels_mapping = {
    0: 'healthy',
    1: 'rust (mild)',
    2: 'rust (mild)',
    3: 'rust (extreme)',
    4: 'rust (extreme)',
}

output_count = 0
for output in outputs:
    extracted_output = output[output.find("[") + 1:output.find("]")]

    output_count += 1

    annotation_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "ann"))
    annotation_paths = glob.glob(os.path.join(annotation_dir, "**", "*.json"), recursive=True)

    annotations = []

for annotation_path in annotation_paths:
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
        annotations.append(annotation)

output_count = 0
print("\n\n --- Results --- \n\n")
for output, annotation in zip(outputs, annotations):
    extracted_output = output[output.find("[") + 1:output.find("]")]
    print("Decision #" + str(output_count) + ": " + extracted_output)
    
    # Extract the classTitle fields
    class_titles = list(find_in_dict('classTitle', annotation))

    # Check if there are multiple classTitle fields
    if len(class_titles) > 1:
        # If there are multiple classTitle fields, determine the appropriate rust level
        rust_level = max(int(class_title.split('_')[-1]) if 'rust_level' in class_title else 0 for class_title in class_titles)
    else:
        # If there is only one classTitle field, determine the rust level as before
        rust_level = int(class_titles[0].split('_')[-1]) if 'rust_level' in class_titles[0] else 0

    expected_output = rust_levels_mapping[rust_level]
    
    # Compare the expected output with the actual output
    if expected_output == extracted_output:
        print("The LLM's output matches the expected output.")
    else:
        print(f"The LLM's output ({extracted_output}) does not match the expected output ({expected_output}).")

    output_count += 1