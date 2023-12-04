import glob
import json
import os
import sys

import replicate

import colorama
from colorama import Fore, Style

use_api = False # Determines if the API will be called or not
outputs_file = "example_outputs.txt" # If use_api is False, this file will be used to get the outputs from

# --- Setup ----

# Used to find all the classTitle fields in the annotation file
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

# Get images
image_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "img"))
image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True))


# Get paths to annotations
annotation_dir = os.path.abspath(os.path.join(script_dir, "..", "ImageData", "ann"))
annotation_paths = sorted(glob.glob(os.path.join(annotation_dir, "**", "*.json"), recursive=True))

# Load annotations
annotations = []
for annotation_path in annotation_paths:
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
        annotations.append(annotation)

print("Amount of images: " + str(len(image_paths)))
print("Amount of annotations: " + str(len(annotations)))
if(len(image_paths) != len(annotations)):
    print(Fore.LIGHTRED_EX + "The amount of images does not match the amount of annotations.")
    print(Style.RESET_ALL)
    sys.exit()

outputs = []
output_annotation_map = {}
if use_api:

    # --- Prepare data ---
    
    prompt = "Are there any diseases that can you detect in this image?, your options are: rust (mild)., rust (severe)., red spider mite., healthy.. You are a part of an automatic farming control loop, therefore a final answer is absolutely necessary. Output your final answer in these brackets: []. It does not matter how sure you are about your answer, it is crucial that you make your best guess anyway. An example of a good output: Based on the image, the leaf appears to have yellow spots in a localised area. The presence of yellow spots on the leaf indicates that the plant is experiencing stress or damage due to rust, the affected area seems small compared to the size of the leaf. Therefore, the final answer is: [rust (mild)]. Another good example: Based on the image, the leaf appears to show signs of bronze discoloration. This is a characteristic sign of an infestation by red spider mites. Red spider mites can feed on a wide variety of plants, causing damage by piercing the plant cells and sucking out the sap. This can cause the leaves to turn yellow or bronze. Therefore, based on these observations, the final answer is: [red spider mites]."

    print("Amount of requests: " + str(len(image_paths)))
    print("Estimated run time: " + str(len(image_paths) * 3 + 8) + " sec.")

    # --- Get outputs from the LVLM ---
    image_count = 0
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

    # Write outputs to file
    outputs_path = os.path.abspath(os.path.join(script_dir, "..", "Outputs", outputs_file))
    with open (outputs_path, "w") as outputs_file:
        for output in outputs:
            outputs_file.write(output + "\n\n")
            outputs_file.seek(-1, os.SEEK_END)
            outputs_file.truncate()

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

# Bind annotation paths to outputs
output_count = 0
for output in outputs:
    output_annotation_map[output] = annotation_paths[output_count]
    output_count += 1

# --- Analyse outputs ---

disease_mapping = {
    0: 'healthy',
    1: 'rust (mild)',
    2: 'rust (mild)',
    3: 'rust (severe)',
    4: 'rust (severe)',
    5: 'red spider mite',
}

# Compare the expected output with the actual output
output_count = 0
correct_count = 0
unsure_count = 0
print("\n\n --- Results --- \n\n")
for output in outputs:
    extracted_output = output[output.find("[") + 1:output.find("]")]
    if not "[" in output or not "]" in output:
        extracted_output = "No decision made"

    annotation_path = output_annotation_map[output]

    print("Decision #" + str(output_count) + ": " + extracted_output)
    print("Image path: " + output_annotation_map[output])
    print("Annotation path: " + annotation_path)

    # Extract the classTitle fields
    class_titles = list(find_key_in_dictionary('classTitle', annotation))

    # Check if there are multiple classTitle fields
    if len(class_titles) > 1:
        for class_title in class_titles:
            print("Class title: " + class_title)
            if "disease_level" in class_title:
                disease_level = int(class_title.split("_")[-1])
            if "red_spider_mite" in class_title:
                disease_level = 5
    else:
        disease_level = 0
    
    if extracted_output == "No decision made":
        print(Fore.LIGHTRED_EX + "The LLM did not make a decision." + Style.RESET_ALL)
        unsure_count += 1
        output_count += 1
        continue

    expected_output = disease_mapping[disease_level]

    # Compare the expected output with the actual output
    if expected_output == extracted_output:
        print(Fore.LIGHTGREEN_EX + "The LLM's output matches the expected output.")
        correct_count += 1
    else:
        print(Fore.LIGHTRED_EX + f"The LLM's output ({extracted_output}) does not match the expected output ({expected_output}).")
    
    print(Style.RESET_ALL)

    output_count += 1

print("\n\n --- Summary --- \n\n")
print("Amount of decisions: " + str(output_count))
print("Amount of correct decisions: " + str(correct_count))
print("Amount of unsure decisions: " + str(unsure_count))
print("Accuracy: " + str(correct_count / output_count * 100) + "%")
print("\n")
