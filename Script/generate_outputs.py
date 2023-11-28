import glob
import json
import os

import replicate

import colorama
from colorama import Fore, Style

use_api = True # Determines if the API will be called or not
outputs_file = "last_outputs.txt" # If use_api is False, this file will be used to get the outputs from

# --- Setup ----

def find_key_in_dictionary(key, dictionary):
    if isinstance(dictionary, dict):
        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for result in find_key_in_dictionary(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in find_key_in_dictionary(key, d):
                        yield result
    elif isinstance(dictionary, list):
        for d in dictionary:
            for result in find_key_in_dictionary(key, d):
                yield result

# Get api key from file
script_dir = os.path.dirname(os.path.abspath(__file__))
key_file_path = os.path.abspath(os.path.join(script_dir, "..", "..", "key.txt")) # Path to the API key file (located in the directory outside of the repository)
with open(key_file_path, "r") as key_file:
    api_key = key_file.read()

os.environ["REPLICATE_API_TOKEN"] = str(api_key)

outputs = []
if use_api:

    # --- Prepare data ---
    
    prompts = ["You are an expert in plant disease detection on leaves, are there any diseases that can you detect in this image?, your options are: rust (mild)., rust (extreme)., healthy.. You are a part of an automatic farming control loop, therefore a final answer is absolutely necessary. Your job is to reason about the health status about the leaf that is provided. Output your final answer in these brackets: []. It does not matter how sure you are about your answer, it is crucial that you make your best guess anyway. An example of a good output: Based on the image, the leaf appears to have yellow spots in a localised area. The presence of yellow spots on the leaf indicates that the plant is experiencing stress or damage due to rust, the affected area seems small compared to the size of the leaf. Therefore, the final answer is: [rust (mild)]",
                "Are there any diseases that can you detect in this image?, your options are: rust (mild)., rust (extreme)., healthy.. You are a part of an automatic farming control loop, therefore a final answer is absolutely necessary. Output your final answer in these brackets: []. It does not matter how sure you are about your answer, it is crucial that you make your best guess anyway. An example of a good output: Based on the image, the leaf appears to have yellow spots in a localised area. The presence of yellow spots on the leaf indicates that the plant is experiencing stress or damage due to rust, the affected area seems small compared to the size of the leaf. Therefore, the final answer is: [rust (mild)]",
              ]

    print("Amount of prompts: " + str(len(prompts)))

    # Get images
    image_paths = glob.glob(os.path.join(script_dir, "..", "ImageData", "img", "**", "*.jpeg"), recursive=True)

    print("Amount of requests: " + str(len(image_paths) * len(prompts)))
    print("Estimated run time: " + str(len(image_paths) * len(prompts) * 5) + " sec.")

    # --- Get outputs from the LVLM ---
    for prompt in prompts:
        image_count = 0
        print("--- Prompt: " + prompt + " ---")

        for image_path in image_paths:
            print("\n\nImage #" + str(image_count) +": "  + image_path)
            with open(image_path, "rb") as image_file:

                output = replicate.run(
                    "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
                    input={"prompt": prompt,
                    "image": image_file}
                )

                for item in output:
                    outputs.append(item)
                    print(item, end="")

                image_count += 1

        print("\n")
    
    # --- Save outputs to file ---

    with open(os.path.join(script_dir, "..", "Outputs", outputs_file), "w") as outputs_file:
        for output in outputs:
            outputs_file.write(output + "\n\n")

else: # Not using the API

    # Get outputs from file
    outputs_path = os.path.abspath(os.path.join(script_dir, "..", "Outputs", "example_outputs.txt"))
    with open (outputs_path, "r") as outputs_file:
        outputs_text = outputs_file.read()
        outputs = outputs_text.split("\n\n")

    output_count = 0
    for output in outputs:
        print("Output #" + str(output_count) + ": " + output + "\n")
        output_count += 1

# --- Analyse outputs ---

rust_levels_mapping = {
    0: 'healthy',
    1: 'rust (mild)',
    2: 'rust (mild)',
    3: 'rust (extreme)',
    4: 'rust (extreme)',
}

# Get paths to annotations
annotation_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "ann"))
annotation_paths = glob.glob(os.path.join(annotation_dir, "**", "*.json"), recursive=True)

# Load annotations
annotations = []
for annotation_path in annotation_paths:
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
        annotations.append(annotation)

# Compare the expected output with the actual output
output_count = 0
correct_count = 0
print("\n --- Results --- \n")
for output, annotation in zip(outputs, annotations):
    if len(outputs) % len(annotations) == 0 and use_api:
        print("--- Prompt: " + prompts[int(output_count / image_count)] + " ---")
    
    if output.find("[") == -1 or output.find("]") == -1:
        print(Fore.LIGHTRED_EX + "The LLM's output does not contain the expected output." + Style.RESET_ALL)
        output_count += 1
        continue
    extracted_output = output[output.find("[") + 1:output.find("]")]
    print("Decision #" + str(output_count) + ": " + extracted_output)
    
    # Extract the classTitle fields
    class_titles = list(find_key_in_dictionary('classTitle', annotation))

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
        print(Fore.LIGHTGREEN_EX + "The LLM's output matches the expected output." + Style.RESET_ALL)
        correct_count += 1
    else:
        print(Fore.LIGHTRED_EX + f"The LLM's output ({extracted_output}) does not match the expected output ({expected_output})." + Style.RESET_ALL)

    output_count += 1

    if len(outputs) % len(annotations) == 0 and use_api:
        print("\n --- Summary --- \n")
        print("Amount of decisions: " + str(output_count))
        print("Amount of correct decisions: " + str(correct_count))
        print("Accuracy: " + str(correct_count / output_count * 100) + "%")